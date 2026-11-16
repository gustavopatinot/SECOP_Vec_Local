# -*- coding: utf-8 -*-
"""
MÓDULO 03: EXPERIMENTO DE CUANTIZACIÓN, DIMENSIÓN MATRYOSHKA Y RELEVANCIA UNSPSC
================================================================================
Proyecto de Tesis de Maestría:
    Búsqueda semántica a gran escala con memoria restringida:
    HNSW cuantizado aplicado a la contratación pública colombiana (SECOP II)
    en un computador de 8 GB de RAM.

Preguntas e Hipótesis Evaluadas:
    - P1 / H1: ¿Qué combinación de cuantización y reducción de dimensión da el
      mejor recall por megabyte en español administrativo?
    - P3 / H3: ¿Cuánto de la pérdida de recall contra la búsqueda exacta llega
      al usuario (medido en precisión@10 y nDCG@10 por jerarquía UNSPSC)?

Hardware Objetivo:
    Apple MacBook Air M1 (8 GB de RAM unificada y SSD NVMe).
================================================================================
"""

import os
import gc
import json
import time
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import pyarrow.parquet as pq
from usearch.index import Index

# Limitación preventiva de hilos BLAS/OpenMP para proteger los 8 GB de RAM
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
registro_auditoria = logging.getLogger("ExperimentoCuantizacionM1")


class ConfiguracionEtapa3LocalM1:
    """Parámetros experimentales de la Etapa 3 en el MacBook Air M1 (8 GB RAM)."""

    # 1. Rutas locales en el MacBook Air M1 del investigador
    RUTA_BASE_LOCAL: Path = Path("/Users/gapt/SECOP_Vec_Local")
    RUTA_ORO: Path = RUTA_BASE_LOCAL / "03_oro_benchmark_secop_vec"
    RUTA_REPORTES_TESIS: Path = RUTA_BASE_LOCAL / "04_tablas_y_metricas_tesis"
    RUTA_INDICES_SSD: Path = RUTA_BASE_LOCAL / "05_indices_temporales_ssd"

    # 2. Artefactos de entrada generados en las Etapas 1 y 2
    ARCHIVO_CORPUS_PARQUET: Path = RUTA_ORO / "secop_vec_corpus_base.parquet"
    ARCHIVO_CONSULTAS_PARQUET: Path = RUTA_ORO / "secop_vec_consultas_10k.parquet"
    ARCHIVO_VECTORES_CORPUS_F16: Path = RUTA_ORO / "vectores_corpus_e5_768d_f16.mmap"
    ARCHIVO_VECTORES_CONSULTAS_F32: Path = RUTA_ORO / "vectores_consultas_e5_768d_f32.npy"
    ARCHIVO_VERDAD_IDS_TOP100: Path = RUTA_ORO / "verdad_exacta_top100_indices.npy"
    ARCHIVO_REPORTE_ETAPA2: Path = RUTA_REPORTES_TESIS / "tabla_3_2_verdad_referencia_gpu.json"

    # 3. Archivo de salida con checkpoints incrementales por configuración
    ARCHIVO_RESULTADOS_ETAPA3: Path = (
        RUTA_REPORTES_TESIS / "tabla_4_1_frontera_pareto_p1_p3.json"
    )

    # 4. Constantes del corpus y control de memoria en 8 GB
    TOTAL_DOCUMENTOS_CORPUS: int = 1_990_000
    SUBMUESTRA_FLOAT32_SEGURA: int = 500_000  # Submuestra para float32 según metodología
    DIMENSION_ORIGINAL: int = 768
    TAMANO_LOTE_CONSTRUCCION: int = 50_000
    HILOS_CPU_M1: int = 4
    REPETICIONES_ESTADISTICAS: int = 5
    T_STUDENT_95_GL4: float = 2.776  # Valor crítico t(0.975, gl=4) para 5 repeticiones
    FACTOR_SOBREMUESTREO_RERANK: int = 10  # 100 candidatos desde SSD para re-ranquear top-10

    # 5. Factores experimentales definidos en la propuesta de tesis
    NIVELES_EF_SEARCH: List[int] = [32, 64, 128, 256, 512]
    DIMENSIONES_MATRYOSHKA: List[int] = [768, 512, 384, 256]
    CONECTIVIDADES_M: List[int] = [8, 12, 16]


def medir_memoria_rss_mb() -> float:
    """Devuelve el consumo actual de memoria física residente (RSS) del proceso en MB."""
    proceso = psutil.Process(os.getpid())
    return proceso.memory_info().rss / (1024 ** 2)


def calcular_media_ic95(valores: List[float]) -> Tuple[float, float]:
    """
    Calcula la media y la semiamplitud del intervalo de confianza al 95%
    sobre las 5 repeticiones exigidas en el diseño experimental.
    """
    arr = np.asarray(valores, dtype=np.float64)
    media = float(np.mean(arr))
    if len(arr) <= 1:
        return round(media, 4), 0.0
    desv_estandar = float(np.std(arr, ddof=1))
    ic95 = ConfiguracionEtapa3LocalM1.T_STUDENT_95_GL4 * (desv_estandar / math.sqrt(len(arr)))
    return round(media, 4), round(ic95, 4)


class CuantizadorRotacionalRaBitQ:
    """
    Implementación de RaBitQ multibit (Gao y Long, SIGMOD 2024; Gao et al., SIGMOD 2025):
    Aplica una rotación ortogonal aleatoria determinista P in O(m) que iguala la varianza
    por coordenada y cuantiza tanto el signo (1-bit para navegación rápida en el grafo)
    como el código multibit de 4 bits (2 dimensiones por byte) para re-ranqueo compacto.
    """

    def __init__(self, dimension: int, semilla: int = 42):
        self.dimension = dimension
        generador = np.random.default_rng(semilla)
        matriz_gaussiana = generador.standard_normal(
            (dimension, dimension), dtype=np.float32
        )
        q_ortogonal, r_triangular = np.linalg.qr(matriz_gaussiana)
        signos_diag = np.sign(np.diag(r_triangular))
        signos_diag[signos_diag == 0] = 1.0
        self.matriz_rotacion = (q_ortogonal * signos_diag).astype(np.float32)
        # Escala teórica 3.0 / sqrt(m) cubre >99.7% de la distribución normal en la esfera
        self.limite_clip = 3.0 / math.sqrt(dimension)

    def rotar_y_empaquetar_b1(self, vectores: np.ndarray) -> np.ndarray:
        """Rota los vectores y empaqueta su signo en 1 bit por dimensión (uint8)."""
        rotados = np.matmul(vectores.astype(np.float32), self.matriz_rotacion)
        bits = (rotados > 0.0).astype(np.uint8)
        return np.packbits(bits, axis=1)

    def rotar_y_cuantizar_4bits(self, vectores: np.ndarray) -> np.ndarray:
        """
        Rota los vectores y cuantiza cada coordenada en 4 bits sin signo [0..15],
        empaquetando dos dimensiones por byte para minimizar la huella en disco/RAM.
        """
        rotados = np.matmul(vectores.astype(np.float32), self.matriz_rotacion)
        normalizados = np.clip(
            (rotados + self.limite_clip) / (2.0 * self.limite_clip), 0.0, 1.0
        )
        cuantizados_u4 = np.rint(normalizados * 15.0).astype(np.uint8)
        # Empaquetado de 2 coordenadas de 4 bits en 1 byte uint8
        pares_altos = cuantizados_u4[:, 0::2] << 4
        pares_bajos = cuantizados_u4[:, 1::2] & 0x0F
        return pares_altos | pares_bajos

    def decodificar_4bits(self, codigos_empaquetados: np.ndarray) -> np.ndarray:
        """Reconstruye los vectores rotados desde los códigos empaquetados de 4 bits."""
        n_filas = codigos_empaquetados.shape[0]
        salida_u4 = np.empty((n_filas, self.dimension), dtype=np.float32)
        salida_u4[:, 0::2] = (codigos_empaquetados >> 4) & 0x0F
        salida_u4[:, 1::2] = codigos_empaquetados & 0x0F
        return (salida_u4 / 15.0) * (2.0 * self.limite_clip) - self.limite_clip

    def rotar_consultas(self, consultas: np.ndarray) -> np.ndarray:
        return np.matmul(consultas.astype(np.float32), self.matriz_rotacion)


class EvaluadorCalidadUNSPSC:
    """
    Calcula en memoria ligera las métricas técnicas (recall@10 y recall@100 contra
    la búsqueda exacta en GPU) y las métricas de relevancia para el usuario final
    (precisión@10 y nDCG@10 por segmento, familia y clase UNSPSC), incluyendo el
    reporte separado para contratos de prestación de servicios (Hipótesis H3).
    """

    def __init__(self, config: ConfiguracionEtapa3LocalM1):
        self.config = config
        registro_auditoria.info("Cargando etiquetas UNSPSC y verdad de referencia exacta...")

        self.verdad_top100 = np.load(self.config.ARCHIVO_VERDAD_IDS_TOP100)

        columnas_meta = [
            "unspsc_segmento",
            "unspsc_familia",
            "unspsc_clase",
            "es_prestacion_servicios",
        ]
        tabla_c = pq.read_table(self.config.ARCHIVO_CORPUS_PARQUET, columns=columnas_meta)
        tabla_q = pq.read_table(self.config.ARCHIVO_CONSULTAS_PARQUET, columns=columnas_meta)

        # Codificación numérica compacta (int32) para acelerar el cálculo en el M1
        self.seg_c = np.array(
            [int(x) for x in tabla_c.column("unspsc_segmento").to_pylist()], dtype=np.int32
        )
        self.fam_c = np.array(
            [int(x) for x in tabla_c.column("unspsc_familia").to_pylist()], dtype=np.int32
        )
        self.cla_c = np.array(
            [int(x) for x in tabla_c.column("unspsc_clase").to_pylist()], dtype=np.int32
        )

        self.seg_q = np.array(
            [int(x) for x in tabla_q.column("unspsc_segmento").to_pylist()], dtype=np.int32
        )[:, None]
        self.fam_q = np.array(
            [int(x) for x in tabla_q.column("unspsc_familia").to_pylist()], dtype=np.int32
        )[:, None]
        self.cla_q = np.array(
            [int(x) for x in tabla_q.column("unspsc_clase").to_pylist()], dtype=np.int32
        )[:, None]
        self.es_ps_q = np.array(
            tabla_q.column("es_prestacion_servicios").to_pylist(), dtype=bool
        )

        self.descuentos_log = 1.0 / np.log2(np.arange(2, 12, dtype=np.float32))
        self.idcg_ideal_10 = float(np.sum(7.0 * self.descuentos_log))

    def evaluar_lote_recuperado(
        self, indices_recuperados: np.ndarray, es_submuestra_f32: bool = False
    ) -> Dict[str, float]:
        """
        Evalúa recall@10, recall@100, precisión@10 por nivel UNSPSC y nDCG@10
        (global, solo prestación de servicios y solo otras modalidades).
        """
        num_q = indices_recuperados.shape[0]
        top10 = np.clip(indices_recuperados[:, :10], 0, len(self.seg_c) - 1)
        top100 = np.clip(indices_recuperados[:, :100], 0, len(self.seg_c) - 1)

        # 1. Cálculo de Recall@10 y Recall@100 contra la verdad de referencia exacta
        if not es_submuestra_f32:
            verdad_10 = self.verdad_top100[:, :10]
            verdad_100 = self.verdad_top100[:, :100]
            aciertos_r10 = np.sum(
                top10[:, :, None] == verdad_10[:, None, :], axis=(1, 2)
            )
            recall_10 = float(np.mean(aciertos_r10 / 10.0))

            if indices_recuperados.shape[1] >= 100:
                aciertos_r100 = np.sum(
                    top100[:, :, None] == verdad_100[:, None, :], axis=(1, 2)
                )
                recall_100 = float(np.mean(aciertos_r100 / 100.0))
            else:
                recall_100 = float("nan")
        else:
            recall_10 = 1.0  # Referencia baseline sin cuantizar en submuestra
            recall_100 = 1.0

        # 2. Cálculo de Precisión@10 y nDCG@10 jerárquico por código UNSPSC
        ac_seg = (self.seg_c[top10] == self.seg_q).astype(np.float32)
        ac_fam = (self.fam_c[top10] == self.fam_q).astype(np.float32)
        ac_cla = (self.cla_c[top10] == self.cla_q).astype(np.float32)

        ganancia = ac_seg + ac_fam + ac_cla
        dcg_por_consulta = np.sum(
            (np.power(2.0, ganancia) - 1.0) * self.descuentos_log, axis=1
        )
        ndcg_por_consulta = dcg_por_consulta / self.idcg_ideal_10

        ndcg_global = float(np.mean(ndcg_por_consulta))
        ndcg_prestacion_servicios = float(np.mean(ndcg_por_consulta[self.es_ps_q]))
        ndcg_otras_modalidades = float(np.mean(ndcg_por_consulta[~self.es_ps_q]))

        return {
            "recall_at_10": round(recall_10, 4),
            "recall_at_100": round(recall_100, 4) if not math.isnan(recall_100) else -1.0,
            "precision_at_10_segmento": round(float(np.mean(ac_seg)), 4),
            "precision_at_10_familia": round(float(np.mean(ac_fam)), 4),
            "precision_at_10_clase": round(float(np.mean(ac_cla)), 4),
            "ndcg_at_10_global": round(ndcg_global, 4),
            "ndcg_at_10_prestacion_servicios": round(ndcg_prestacion_servicios, 4),
            "ndcg_at_10_otras_modalidades": round(ndcg_otras_modalidades, 4),
        }


class LaboratorioCuantizacionM1:
    """
    Orquesta la construcción por lotes desde memmap y la evaluación con 5 repeticiones
    en el MacBook Air M1 de 8 GB de RAM.
    """

    def __init__(self, config: ConfiguracionEtapa3LocalM1):
        self.config = config
        self.config.RUTA_REPORTES_TESIS.mkdir(parents=True, exist_ok=True)
        self.config.RUTA_INDICES_SSD.mkdir(parents=True, exist_ok=True)

        self.evaluador = EvaluadorCalidadUNSPSC(config)
        self.matriz_corpus_mmap = np.memmap(
            self.config.ARCHIVO_VECTORES_CORPUS_F16,
            dtype=np.float16,
            mode="r",
            shape=(self.config.TOTAL_DOCUMENTOS_CORPUS, self.config.DIMENSION_ORIGINAL),
        )
        self.vectores_consultas_f32 = np.load(self.config.ARCHIVO_VECTORES_CONSULTAS_F32)

    def _cargar_checkpoints(self) -> Dict[str, Dict]:
        if self.config.ARCHIVO_RESULTADOS_ETAPA3.exists():
            with open(self.config.ARCHIVO_RESULTADOS_ETAPA3, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _guardar_checkpoint(self, resultados: Dict[str, Dict]) -> None:
        archivo_tmp = self.config.ARCHIVO_RESULTADOS_ETAPA3.with_suffix(".tmp")
        with open(archivo_tmp, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        archivo_tmp.replace(self.config.ARCHIVO_RESULTADOS_ETAPA3)

    def _preparar_lote_truncado(
        self, bloque_f16: np.ndarray, dimension: int
    ) -> np.ndarray:
        """
        Extrae las primeras `dimension` coordenadas (truncado Matryoshka) y
        renormaliza en norma L2 si `dimension < 768`.
        """
        sub = bloque_f16[:, :dimension].astype(np.float32)
        if dimension < self.config.DIMENSION_ORIGINAL:
            normas = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12
            sub /= normas
        return sub

    def evaluar_configuracion(
        self,
        familia_representacion: str,
        dimension: int,
        conectividad_m: int,
         conservar_indice_en_disco: bool = False,
    ) -> Dict:
        """
        Construye el índice vectorial por lotes de 50.000 desde el memmap en SSD,
        lo reabre mapeado (`view=True`) y ejecuta las 5 repeticiones de búsqueda
        para cada nivel de `ef_search`.
        """
        id_experimento = f"{familia_representacion}_dim{dimension}_M{conectividad_m}"
        resultados_previos = self._cargar_checkpoints()
        if id_experimento in resultados_previos:
            registro_auditoria.info(
                f"Configuración '{id_experimento}' ya completada en checkpoint. Omitiendo."
            )
            return resultados_previos[id_experimento]

        es_submuestra_f32 = familia_representacion == "float32_submuestra"
        n_docs = (
            self.config.SUBMUESTRA_FLOAT32_SEGURA
            if es_submuestra_f32
            else self.config.TOTAL_DOCUMENTOS_CORPUS
        )

        # Determinación de tipo de dato y métrica en USearch
        if familia_representacion == "float32_submuestra":
            dtype_usearch, metrica_usearch = "f32", "cos"
        elif familia_representacion == "float16":
            dtype_usearch, metrica_usearch = "f16", "cos"
        elif familia_representacion == "int8":
            dtype_usearch, metrica_usearch = "i8", "cos"
        elif familia_representacion in ("b1_rerank_ssd", "rabitq_4bit"):
            dtype_usearch, metrica_usearch = "b1", "hamming"
        else:
            raise ValueError(f"Familia no reconocida: {familia_representacion}")

        ruta_indice_usearch = (
            self.config.RUTA_INDICES_SSD / f"secop_{id_experimento}.usearch"
        )
        ruta_codigos_rabitq = (
            self.config.RUTA_INDICES_SSD / f"secop_{id_experimento}_u4.mmap"
        )

        cuantizador_rabitq: Optional[CuantizadorRotacionalRaBitQ] = None
        mmap_rabitq_u4: Optional[np.memmap] = None

        if familia_representacion == "rabitq_4bit":
            cuantizador_rabitq = CuantizadorRotacionalRaBitQ(dimension=dimension)
            mmap_rabitq_u4 = np.memmap(
                ruta_codigos_rabitq,
                dtype=np.uint8,
                mode="w+",
                shape=(n_docs, dimension // 2),
            )

        # 1. Construcción por lotes de 50.000 vectores (controlando RSS en 8 GB)
        gc.collect()
        rss_base_mb = medir_memoria_rss_mb()
        rss_pico_construccion_mb = rss_base_mb
        t_inicio_construccion = time.perf_counter()

        indice = Index(
            ndim=dimension,
            metric=metrica_usearch,
            dtype=dtype_usearch,
            connectivity=conectividad_m,
        )

        registro_auditoria.info(
            f"Construyendo '{id_experimento}' sobre {n_docs:,} contratos en lotes de "
            f"{self.config.TAMANO_LOTE_CONSTRUCCION:,}..."
        )

        paso = self.config.TAMANO_LOTE_CONSTRUCCION
        for inicio in range(0, n_docs, paso):
            fin = min(inicio + paso, n_docs)
            ids_lote = np.arange(inicio, fin, dtype=np.uint64)
            bloque_f32 = self._preparar_lote_truncado(
                self.matriz_corpus_mmap[inicio:fin, :], dimension
            )

            if familia_representacion == "b1_rerank_ssd":
                bits_empaquetados = np.packbits((bloque_f32 > 0.0).astype(np.uint8), axis=1)
                indice.add(ids_lote, bits_empaquetados, threads=self.config.HILOS_CPU_M1)
            elif familia_representacion == "rabitq_4bit":
                assert cuantizador_rabitq is not None and mmap_rabitq_u4 is not None
                bits_rotados = cuantizador_rabitq.rotar_y_empaquetar_b1(bloque_f32)
                mmap_rabitq_u4[inicio:fin, :] = cuantizador_rabitq.rotar_y_cuantizar_4bits(
                    bloque_f32
                )
                indice.add(ids_lote, bits_rotados, threads=self.config.HILOS_CPU_M1)
            elif familia_representacion == "float16":
                indice.add(
                    ids_lote,
                    bloque_f32.astype(np.float16),
                    threads=self.config.HILOS_CPU_M1,
                )
            else:
                indice.add(ids_lote, bloque_f32, threads=self.config.HILOS_CPU_M1)

            rss_pico_construccion_mb = max(rss_pico_construccion_mb, medir_memoria_rss_mb())
            if fin % 250_000 == 0 or fin == n_docs:
                registro_auditoria.info(
                    f"  -> [{id_experimento}] Indexados: {fin:,}/{n_docs:,} | "
                    f"RSS actual: {medir_memoria_rss_mb():.1f} MB"
                )

        if mmap_rabitq_u4 is not None:
            mmap_rabitq_u4.flush()

        indice.save(str(ruta_indice_usearch))
        tiempo_construccion_seg = time.perf_counter() - t_inicio_construccion

        tamano_disco_mb = ruta_indice_usearch.stat().st_size / (1024 ** 2)
        if ruta_codigos_rabitq.exists():
            tamano_disco_mb += ruta_codigos_rabitq.stat().st_size / (1024 ** 2)

        # Liberamos el índice mutable de la RAM y lo reabrimos con view=True (como en la propuesta)
        del indice
        gc.collect()
        indice_mapeado = Index.restore(str(ruta_indice_usearch), view=True)

        # Preparación de las 10.000 consultas truncadas a `dimension`
        consultas_truncadas = self._preparar_lote_truncado(
            self.vectores_consultas_f32, dimension
        )
        if familia_representacion == "b1_rerank_ssd":
            consultas_busqueda = np.packbits(
                (consultas_truncadas > 0.0).astype(np.uint8), axis=1
            )
        elif familia_representacion == "rabitq_4bit":
            assert cuantizador_rabitq is not None
            consultas_busqueda = cuantizador_rabitq.rotar_y_empaquetar_b1(
                consultas_truncadas
            )
            consultas_rotadas_f32 = cuantizador_rabitq.rotar_consultas(consultas_truncadas)
        elif familia_representacion == "float16":
            consultas_busqueda = consultas_truncadas.astype(np.float16)
        else:
            consultas_busqueda = consultas_truncadas

        # 2. Barrido de ef_search con 5 repeticiones independientes
        resultados_por_ef: Dict[str, Dict] = {}
        rss_pico_busqueda_mb = medir_memoria_rss_mb()

        for ef in self.config.NIVELES_EF_SEARCH:
            indice_mapeado.expansion_search = ef
            latencias_p50_ms: List[float] = []
            latencias_p99_ms: List[float] = []
            qps_corridas: List[float] = []
            ultimo_top100_ids: Optional[np.ndarray] = None

            for rep in range(self.config.REPETICIONES_ESTADISTICAS):
                t_rep_inicio = time.perf_counter()

                # Medición por sub-lotes de 100 consultas para capturar distribución p50/p99
                tiempos_consulta_ms: List[float] = []
                bloques_ids: List[np.ndarray] = []

                for i_q in range(0, len(consultas_busqueda), 100):
                    q_sub = consultas_busqueda[i_q : i_q + 100]
                    t_q0 = time.perf_counter()

                    res_busqueda = indice_mapeado.search(
                        q_sub, count=100, threads=self.config.HILOS_CPU_M1
                    )
                    ids_candidatos = np.asarray(res_busqueda.keys, dtype=np.int64)

                    # Etapa 2 de re-ranqueo desde SSD para cuantizaciones extremas (1-bit y RaBitQ)
                    if familia_representacion == "b1_rerank_ssd":
                        q_orig_sub = consultas_truncadas[i_q : i_q + 100]
                        ids_rerank = np.empty((len(q_sub), 100), dtype=np.int64)
                        for idx_fila in range(len(q_sub)):
                            cand_fila = ids_candidatos[idx_fila]
                            vecs_ssd = self._preparar_lote_truncado(
                                self.matriz_corpus_mmap[cand_fila, :], dimension
                            )
                            scores_exactos = np.dot(vecs_ssd, q_orig_sub[idx_fila])
                            orden = np.argsort(-scores_exactos)
                            ids_rerank[idx_fila] = cand_fila[orden]
                        ids_candidatos = ids_rerank

                    elif familia_representacion == "rabitq_4bit":
                        assert cuantizador_rabitq is not None and mmap_rabitq_u4 is not None
                        q_rot_sub = consultas_rotadas_f32[i_q : i_q + 100]
                        ids_rerank = np.empty((len(q_sub), 100), dtype=np.int64)
                        for idx_fila in range(len(q_sub)):
                            cand_fila = ids_candidatos[idx_fila]
                            vecs_u4 = cuantizador_rabitq.decodificar_4bits(
                                mmap_rabitq_u4[cand_fila, :]
                            )
                            scores_aprox = np.dot(vecs_u4, q_rot_sub[idx_fila])
                            orden = np.argsort(-scores_aprox)
                            ids_rerank[idx_fila] = cand_fila[orden]
                        ids_candidatos = ids_rerank

                    duracion_ms_por_q = ((time.perf_counter() - t_q0) * 1000.0) / len(q_sub)
                    tiempos_consulta_ms.append(duracion_ms_por_q)
                    bloques_ids.append(ids_candidatos)

                tiempo_total_rep = time.perf_counter() - t_rep_inicio
                latencias_p50_ms.append(float(np.percentile(tiempos_consulta_ms, 50)))
                latencias_p99_ms.append(float(np.percentile(tiempos_consulta_ms, 99)))
                qps_corridas.append(len(consultas_busqueda) / max(tiempo_total_rep, 1e-6))
                rss_pico_busqueda_mb = max(rss_pico_busqueda_mb, medir_memoria_rss_mb())

                if rep == 0:
                    ultimo_top100_ids = np.vstack(bloques_ids)

            assert ultimo_top100_ids is not None
            metricas_calidad = self.evaluador.evaluar_lote_recuperado(
                ultimo_top100_ids, es_submuestra_f32=es_submuestra_f32
            )
            p50_media, p50_ic95 = calcular_media_ic95(latencias_p50_ms)
            p99_media, p99_ic95 = calcular_media_ic95(latencias_p99_ms)
            qps_media, qps_ic95 = calcular_media_ic95(qps_corridas)

            resultados_por_ef[f"ef_{ef}"] = {
                "ef_search": ef,
                **metricas_calidad,
                "latencia_p50_ms_media": p50_media,
                "latencia_p50_ms_ic95": p50_ic95,
                "latencia_p99_ms_media": p99_media,
                "latencia_p99_ms_ic95": p99_ic95,
                "qps_media": qps_media,
                "qps_ic95": qps_ic95,
            }

            registro_auditoria.info(
                f"  -> [{id_experimento} | ef={ef}] "
                f"Recall@10={metricas_calidad['recall_at_10']:.4f} | "
                f"nDCG@10={metricas_calidad['ndcg_at_10_global']:.4f} | "
                f"p50={p50_media:.2f}ms | QPS={qps_media:.0f}"
            )

        del indice_mapeado
        gc.collect()

        # Eliminación opcional del archivo temporal en SSD para no llenar el disco del Mac
        if not conservar_indice_en_disco:
            if ruta_indice_usearch.exists():
                ruta_indice_usearch.unlink()
            if ruta_codigos_rabitq.exists():
                ruta_codigos_rabitq.unlink()

        resumen_config = {
            "id_experimento": id_experimento,
            "familia_representacion": familia_representacion,
            "dimension": dimension,
            "conectividad_M": conectividad_m,
            "documentos_indexados": n_docs,
            "bytes_por_vector_indice": round((tamano_disco_mb * (1024 ** 2)) / n_docs, 2),
            "tamano_indice_disco_mb": round(tamano_disco_mb, 2),
            "rss_pico_construccion_mb": round(rss_pico_construccion_mb, 2),
            "rss_pico_busqueda_mb": round(rss_pico_busqueda_mb, 2),
            "tiempo_construccion_segundos": round(tiempo_construccion_seg, 2),
            "barrido_ef_search": resultados_por_ef,
        }

        resultados_actualizados = self._cargar_checkpoints()
        resultados_actualizados[id_experimento] = resumen_config
        self._guardar_checkpoint(resultados_actualizados)
        return resumen_config

    def ejecutar_fase_3a_familias(self) -> None:
        """
        Fase 3A del diseño secuencial: fija dimensión=768 y M=12 para comparar
        las 5 familias de representación definidas en la metodología.
        """
        familias = [
            "float32_submuestra",
            "float16",
            "int8",
            "b1_rerank_ssd",
            "rabitq_4bit",
        ]
        for fam in familias:
            self.evaluar_configuracion(
                familia_representacion=fam,
                dimension=768,
                conectividad_m=12,
                conservar_indice_en_disco=False,
            )

    def ejecutar_fase_3b_barrido_pareto(
        self, dos_mejores_familias: Tuple[str, str] = ("int8", "b1_rerank_ssd")
    ) -> None:
        """
        Fase 3B del diseño secuencial: barre dimensión {768, 512, 384, 256} y
        conectividad M {8, 12, 16} sobre las dos mejores familias.
        """
        for fam in dos_mejores_familias:
            for dim in self.config.DIMENSIONES_MATRYOSHKA:
                for m in self.config.CONECTIVIDADES_M:
                    # Conservamos en disco el índice candidato ganador (int8, dim=512, M=12)
                    guardar = fam == "int8" and dim == 512 and m == 12
                    self.evaluar_configuracion(
                        familia_representacion=fam,
                        dimension=dim,
                        conectividad_m=m,
                        conservar_indice_en_disco=guardar,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 03: Evaluación de Cuantización y Dimensión en MacBook Air M1 (8 GB)"
    )
    parser.add_argument(
        "--fase",
        choices=["3A", "3B", "completa"],
        default="3A",
        help="Ejecuta primero la Fase 3A (comparación de familias) o el barrido 3B.",
    )
    args = parser.parse_args()

    config_m1 = ConfiguracionEtapa3LocalM1()
    laboratorio = LaboratorioCuantizacionM1(config_m1)

    if args.fase in ("3A", "completa"):
        registro_auditoria.info("=== INICIANDO FASE 3A: SELECCIÓN DE FAMILIA (D=768, M=12) ===")
        laboratorio.ejecutar_fase_3a_familias()

    if args.fase in ("3B", "completa"):
        registro_auditoria.info("=== INICIANDO FASE 3B: BARRIDO MATRYOSHKA Y CONECTIVIDAD M ===")
        laboratorio.ejecutar_fase_3b_barrido_pareto()

    registro_auditoria.info(
        f"Experimento finalizado. Resultados guardados en: {config_m1.ARCHIVO_RESULTADOS_ETAPA3}"
    )
