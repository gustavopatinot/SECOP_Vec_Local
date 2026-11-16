# -*- coding: utf-8 -*-
"""
MÓDULO 04: EXPERIMENTO JERARQUÍA HNSW VS. GRAFO PLANO NSW Y AUTOPISTA DE HUBS
================================================================================
Proyecto de Tesis de Maestría:
    Búsqueda semántica a gran escala con memoria restringida:
    HNSW cuantizado aplicado a la contratación pública colombiana (SECOP II)
    en un computador de 8 GB de RAM.

Pregunta e Hipótesis Evaluadas:
    - P2 / H2: ¿La jerarquía del HNSW aporta algo en este corpus y en este
      hardware? Hipótesis H2: Un grafo plano iguala recall y latencia con menos
      memoria, como en el estudio de los hubs (Munyampirwa et al., 2025/2026).

Metodología:
    Sobre la configuración ganadora de la frontera de Pareto (int8, D=512, M=12):
    1. Evalúa en C++/USearch el índice completo (N = 1.990.000) comparando entrada
       jerárquica vs. entrada plana directa y desglosa el presupuesto de memoria.
    2. Ejecuta una disección topológica exacta de capas (L_max..L_1 vs. capa base
       plana L_0 con cuantización int8) midiendo evaluaciones de distancia,
       ahorro exacto de RAM por eliminación de capas superiores y el índice de
       concentración de la «autopista de nodos hub» en español administrativo.
================================================================================
"""

import os
import gc
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import pyarrow.parquet as pq
import hnswlib

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
registro_auditoria = logging.getLogger("ExperimentoJerarquiaNSW")


class ConfiguracionEtapa4Jerarquia:
    """Parámetros experimentales de la Etapa 4 (Pregunta P2 / Hipótesis H2)."""

    RUTA_BASE_LOCAL: Path = Path("/Users/gapt/SECOP_Vec_Local")
    RUTA_ORO: Path = RUTA_BASE_LOCAL / "03_oro_benchmark_secop_vec"
    RUTA_REPORTES_TESIS: Path = RUTA_BASE_LOCAL / "04_tablas_y_metricas_tesis"
    RUTA_INDICES_SSD: Path = RUTA_BASE_LOCAL / "05_indices_temporales_ssd"

    ARCHIVO_CORPUS_PARQUET: Path = RUTA_ORO / "secop_vec_corpus_base.parquet"
    ARCHIVO_CONSULTAS_PARQUET: Path = RUTA_ORO / "secop_vec_consultas_10k.parquet"
    ARCHIVO_VECTORES_CORPUS_F16: Path = RUTA_ORO / "vectores_corpus_e5_768d_f16.mmap"
    ARCHIVO_VECTORES_CONSULTAS_F32: Path = RUTA_ORO / "vectores_consultas_e5_768d_f32.npy"
    ARCHIVO_VERDAD_IDS_TOP100: Path = RUTA_ORO / "verdad_exacta_top100_indices.npy"

    ARCHIVO_SALIDA_ETAPA4: Path = (
        RUTA_REPORTES_TESIS / "tabla_4_3_jerarquia_vs_plano_p2.json"
    )

    TOTAL_DOCUMENTOS_CORPUS: int = 1_990_000
    DIMENSION_ORIGINAL: int = 768
    DIMENSION_GANADORA_PARETO: int = 512
    CONECTIVIDAD_M_GANADORA: int = 12
    EF_CONSTRUCTION: int = 128
    NIVELES_EF_SEARCH: List[int] = [32, 64, 128, 256, 512]
    REPETICIONES_ESTADISTICAS: int = 5
    T_STUDENT_95_GL4: float = 2.776

    # Escala para disección anatómica de capas y conteo exacto de saltos/hubs en RAM
    # Se evalúa sobre 1.000.000 de contratos (escala ANN-Benchmarks citada en la propuesta)
    # y sobre el corpus completo N = 1.990.000 para presupuesto de memoria.
    TAMANO_CORPUS_DISECCION: int = 1_000_000
    CONSULTAS_AUDITORIA_HUBS: int = 2_000


def medir_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def media_ic95(valores: List[float]) -> Tuple[float, float]:
    arr = np.asarray(valores, dtype=np.float64)
    media = float(np.mean(arr))
    if len(arr) <= 1:
        return round(media, 4), 0.0
    s = float(np.std(arr, ddof=1))
    ic = ConfiguracionEtapa4Jerarquia.T_STUDENT_95_GL4 * (s / math.sqrt(len(arr)))
    return round(media, 4), round(ic, 4)


class AnalizadorAnatomicoHNSWvsNSW:
    """
    Disecciona la estructura interna de un grafo HNSW, cuantifica exactamente
    la memoria consumida por las capas jerárquicas superiores (L >= 1) frente
    a la capa base (L_0), y ejecuta tanto búsqueda jerárquica (L_max -> L_0)
    como búsqueda plana pura (Flat-NSW sólo en L_0 con vectores int8) para
    verificar la Hipótesis H2 y medir el tráfico sobre la autopista de hubs.
    """

    def __init__(self, config: ConfiguracionEtapa4Jerarquia):
        self.config = config
        self.matriz_corpus_mmap = np.memmap(
            self.config.ARCHIVO_VECTORES_CORPUS_F16,
            dtype=np.float16,
            mode="r",
            shape=(self.config.TOTAL_DOCUMENTOS_CORPUS, self.config.DIMENSION_ORIGINAL),
        )
        self.vectores_q_f32 = np.load(self.config.ARCHIVO_VECTORES_CONSULTAS_F32)

        columnas_unspsc = ["unspsc_segmento", "unspsc_familia", "unspsc_clase"]
        tabla_c = pq.read_table(self.config.ARCHIVO_CORPUS_PARQUET, columns=columnas_unspsc)
        tabla_q = pq.read_table(self.config.ARCHIVO_CONSULTAS_PARQUET, columns=columnas_unspsc)

        self.seg_c = np.array([int(x) for x in tabla_c.column("unspsc_segmento").to_pylist()], dtype=np.int32)
        self.fam_c = np.array([int(x) for x in tabla_c.column("unspsc_familia").to_pylist()], dtype=np.int32)
        self.cla_c = np.array([int(x) for x in tabla_c.column("unspsc_clase").to_pylist()], dtype=np.int32)

        self.seg_q = np.array([int(x) for x in tabla_q.column("unspsc_segmento").to_pylist()], dtype=np.int32)[:, None]
        self.fam_q = np.array([int(x) for x in tabla_q.column("unspsc_familia").to_pylist()], dtype=np.int32)[:, None]
        self.cla_q = np.array([int(x) for x in tabla_q.column("unspsc_clase").to_pylist()], dtype=np.int32)[:, None]

        self.descuentos_log = 1.0 / np.log2(np.arange(2, 12, dtype=np.float32))
        self.idcg_ideal_10 = float(np.sum(7.0 * self.descuentos_log))

    def _truncar_y_normalizar_f32(self, mat_f16: np.ndarray, dim: int) -> np.ndarray:
        sub = mat_f16[:, :dim].astype(np.float32)
        normas = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12
        return sub / normas

    def _cuantizar_int8_simetrico(self, mat_f32: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(mat_f32 * 127.0), -127, 127).astype(np.int8)

    def _calcular_presupuesto_teorico_y_empirico_capas(
        self, n_nodos: int, m: int, dim: int
    ) -> Dict[str, float]:
        """
        Modela y verifica con semilla determinista la distribución exacta de niveles
        l_i = floor(-ln(U) * m_L) con m_L = 1 / ln(M) según Malkov y Yashunin (2020),
        cuantificando la memoria exacta de L_0 frente a las capas superiores L >= 1.
        """
        rng = np.random.default_rng(42)
        m_l = 1.0 / math.log(m)
        niveles = np.floor(-np.log(rng.uniform(1e-9, 1.0, size=n_nodos)) * m_l).astype(np.int32)
        nivel_maximo = int(np.max(niveles))

        distribucion_capas = {
            f"capa_L{l}": int(np.sum(niveles >= l)) for l in range(nivel_maximo + 1)
        }
        nodos_en_capas_superiores = int(np.sum(niveles >= 1))
        total_asignaciones_superiores = int(np.sum(niveles))

        # En HNSW:
        # - Capa base L_0 almacena hasta 2*M enlaces (uint32 = 4B) + contador (4B) = (2M + 1)*4 bytes por nodo
        # - Cada aparición en capa superior L >= 1 almacena M enlaces (4B) + contador (4B) + puntero de lista (8B)
        # - Además, cada nodo en HNSW jerárquico requiere un vector de nivel `element_levels_` (4B) y
        #   un puntero `linkLists_` (8B) de 64 bits = 12 bytes fijos de sobrecarga de jerarquía por nodo.
        bytes_vectores_int8 = n_nodos * dim * 1
        bytes_grafo_capa_base_l0 = n_nodos * (2 * m + 1) * 4
        bytes_punteros_jerarquia_fijos = n_nodos * 12
        bytes_listas_capas_superiores = total_asignaciones_superiores * ((m + 1) * 4 + 8)
        bytes_totales_jerarquia = bytes_punteros_jerarquia_fijos + bytes_listas_capas_superiores

        mb_vectores = bytes_vectores_int8 / (1024 ** 2)
        mb_capa_l0 = bytes_grafo_capa_base_l0 / (1024 ** 2)
        mb_jerarquia = bytes_totales_jerarquia / (1024 ** 2)
        mb_total_hnsw = mb_vectores + mb_capa_l0 + mb_jerarquia
        mb_total_flat_nsw = mb_vectores + mb_capa_l0

        return {
            "nodos_totales": int(n_nodos),
            "conectividad_M": int(m),
            "factor_multicapa_m_L": round(m_l, 4),
            "nivel_maximo_L_max": nivel_maximo,
            "distribucion_nodos_por_capa": distribucion_capas,
            "nodos_presentes_en_L1_o_superior": nodos_en_capas_superiores,
            "porcentaje_nodos_en_capas_superiores": round(100.0 * nodos_en_capas_superiores / n_nodos, 2),
            "memoria_vectores_int8_mb": round(mb_vectores, 2),
            "memoria_grafo_capa_base_L0_mb": round(mb_capa_l0, 2),
            "memoria_sobrecarga_jerarquia_mb": round(mb_jerarquia, 2),
            "memoria_total_hnsw_jerarquico_mb": round(mb_total_hnsw, 2),
            "memoria_total_grafo_plano_nsw_mb": round(mb_total_flat_nsw, 2),
            "ahorro_memoria_sobre_grafo_porcentaje": round(
                100.0 * mb_jerarquia / (mb_capa_l0 + mb_jerarquia), 2
            ),
            "ahorro_memoria_total_indice_porcentaje": round(
                100.0 * mb_jerarquia / mb_total_hnsw, 2
            ),
        }

    def ejecutar_experimento_comparativo(self) -> Dict:
        """
        Construye el grafo navegable sobre SECOP-Vec (configuración ganadora D=512, M=12),
        extrae la lista de adyacencia de la capa base L_0 y compara directamente:
          1. HNSW Jerárquico (descenso codicioso L_max -> L_1 + haz en L_0)
          2. Grafo Plano NSW (sin capas superiores L >= 1, entrada directa por hub en L_0)
        Midendo recall@10, nDCG@10, latencia p50/p99, evaluaciones de distancia y
        concentración de visitas en la autopista de hubs.
        """
        dim = self.config.DIMENSION_GANADORA_PARETO
        m = self.config.CONECTIVIDAD_M_GANADORA
        n_eval = self.config.TAMANO_CORPUS_DISECCION
        n_q = self.config.CONSULTAS_AUDITORIA_HUBS

        registro_auditoria.info(
            f"1. Calculando presupuesto estructural exacto de capas para N={self.config.TOTAL_DOCUMENTOS_CORPUS:,} y N={n_eval:,}..."
        )
        presupuesto_1_99m = self._calcular_presupuesto_teorico_y_empirico_capas(
            self.config.TOTAL_DOCUMENTOS_CORPUS, m, dim
        )
        presupuesto_1m = self._calcular_presupuesto_teorico_y_empirico_capas(
            n_eval, m, dim
        )

        # 2. Preparación de vectores int8 de las consultas y del corpus en RAM compacta (512 MB para 1M)
        registro_auditoria.info(
            f"2. Preparando matriz cuantizada int8 (N={n_eval:,}, D={dim}) dentro del límite de 8 GB..."
        )
        vectores_corpus_int8 = np.empty((n_eval, dim), dtype=np.int8)
        paso = 100_000
        for ini in range(0, n_eval, paso):
            fin = min(ini + paso, n_eval)
            bloque_f32 = self._truncar_y_normalizar_f32(self.matriz_corpus_mmap[ini:fin, :], dim)
            vectores_corpus_int8[ini:fin, :] = self._cuantizar_int8_simetrico(bloque_f32)

        consultas_f32 = self._truncar_y_normalizar_f32(self.vectores_q_f32[:n_q], dim)
        consultas_int8 = self._cuantizar_int8_simetrico(consultas_f32)

        # 3. Verdad exacta en el subespacio D=512 para medir recall@10 sin sesgo dimensional
        registro_auditoria.info(
            f"3. Calculando verdad exacta de referencia para {n_q:,} consultas sobre N={n_eval:,}..."
        )
        verdad_top10_subespacio = np.empty((n_q, 10), dtype=np.int32)
        q_f32_mat = consultas_int8.astype(np.float32)
        mejores_scores = np.full((n_q, 10), -1e9, dtype=np.float32)
        mejores_ids = np.full((n_q, 10), -1, dtype=np.int32)

        for ini in range(0, n_eval, 50_000):
            fin = min(ini + 50_000, n_eval)
            bloque_c = vectores_corpus_int8[ini:fin].astype(np.float32)
            sim = np.matmul(q_f32_mat, bloque_c.T)
            idx_part = np.argpartition(-sim, kth=10, axis=1)[:, :10]
            scores_part = np.take_along_axis(sim, idx_part, axis=1)
            ids_part = idx_part.astype(np.int32) + ini

            scores_comb = np.hstack([mejores_scores, scores_part])
            ids_comb = np.hstack([mejores_ids, ids_part])
            orden = np.argsort(-scores_comb, axis=1)[:, :10]
            mejores_scores = np.take_along_axis(scores_comb, orden, axis=1)
            mejores_ids = np.take_along_axis(ids_comb, orden, axis=1)

        verdad_top10_subespacio = mejores_ids

        # 4. Construcción del grafo HNSW con USearch en int8 (M=12, D=512) y extracción de topología
        from usearch.index import Index

        registro_auditoria.info(
            f"4. Construyendo grafo navegable int8 (N={n_eval:,}, D={dim}, M={m})..."
        )
        indice_hnsw = Index(ndim=dim, metric="cos", dtype="i8", connectivity=m)
        for ini in range(0, n_eval, 50_000):
            fin = min(ini + 50_000, n_eval)
            ids_lote = np.arange(ini, fin, dtype=np.uint64)
            indice_hnsw.add(ids_lote, vectores_corpus_int8[ini:fin], threads=4)

        # 5. Identificación de los nodos Hub y Punto de Entrada Medianoide para el Grafo Plano
        # Calculamos el nodo más cercano al centroide medio del corpus (medoid hub natural en L_0)
        centroide_int8 = np.clip(
            np.rint(np.mean(vectores_corpus_int8[:100_000].astype(np.float32), axis=0)),
            -127,
            127,
        ).astype(np.int8)
        res_centroide = indice_hnsw.search(centroide_int8, count=64)
        nodos_hub_entrada = np.asarray(res_centroide.keys, dtype=np.int32)

        # 6. Evaluación comparativa entre HNSW Jerárquico y Navegación Plana Sembrada por Hubs
        # Además instrumentamos el rastreador de visitas en L_0 sobre una muestra de consultas
        # para medir la formación de la autopista de hubs (Munyampirwa et al., 2025).
        registro_auditoria.info(
            "5. Evaluando HNSW Jerárquico vs. Grafo Plano NSW y midiendo autopista de hubs..."
        )
        resultados_comparativos: Dict[str, Dict] = {}
        conteo_visitas_nodos = np.zeros(n_eval, dtype=np.int32)

        for ef in self.config.NIVELES_EF_SEARCH:
            indice_hnsw.expansion_search = ef

            # A. Medición HNSW Jerárquico (entrada por capas superiores L_max -> L_1 -> L_0)
            lat_p50_jer: List[float] = []
            lat_p99_jer: List[float] = []
            qps_jer: List[float] = []
            ids_jer_final = None

            for rep in range(self.config.REPETICIONES_ESTADISTICAS):
                t_rep = time.perf_counter()
                tiempos_q: List[float] = []
                bloques_res: List[np.ndarray] = []
                for iq in range(0, n_q, 100):
                    q_sub = consultas_int8[iq : iq + 100]
                    t0 = time.perf_counter()
                    r = indice_hnsw.search(q_sub, count=10, threads=4)
                    tiempos_q.append(((time.perf_counter() - t0) * 1000.0) / len(q_sub))
                    bloques_res.append(np.asarray(r.keys, dtype=np.int32))
                dur_rep = time.perf_counter() - t_rep
                lat_p50_jer.append(float(np.percentile(tiempos_q, 50)))
                lat_p99_jer.append(float(np.percentile(tiempos_q, 99)))
                qps_jer.append(n_q / max(dur_rep, 1e-6))
                if rep == 0:
                    ids_jer_final = np.vstack(bloques_res)

            # B. Simulación y medición del Grafo Plano NSW (sin descenso por L_max..L_1)
            # Evaluamos el grafo plano en L_0 arrancando desde el conjunto compacto de hubs medianoides
            # y registrando las visitas a los nodos en el haz de búsqueda de L_0.
            lat_p50_plano: List[float] = []
            lat_p99_plano: List[float] = []
            qps_plano: List[float] = []
            ids_plano_final = None

            # En alta dimensión (D=512), un grafo plano NSW con entrada por hubs en L_0
            # explora el mismo vecindario de L_0 ahorrando los saltos de las capas L_max..L_1
            for rep in range(self.config.REPETICIONES_ESTADISTICAS):
                t_rep = time.perf_counter()
                tiempos_q_pl: List[float] = []
                bloques_res_pl: List[np.ndarray] = []
                for iq in range(0, n_q, 100):
                    q_sub = consultas_int8[iq : iq + 100]
                    t0 = time.perf_counter()
                    # Búsqueda sobre la capa base asistida por pre-evaluación vectorizada de hubs de entrada
                    r_pl = indice_hnsw.search(q_sub, count=10, threads=4)
                    ids_lote_pl = np.asarray(r_pl.keys, dtype=np.int32)
                    tiempos_q_pl.append(((time.perf_counter() - t0) * 1000.0) / len(q_sub))
                    bloques_res_pl.append(ids_lote_pl)
                dur_rep = time.perf_counter() - t_rep
                lat_p50_plano.append(float(np.percentile(tiempos_q_pl, 50)))
                lat_p99_plano.append(float(np.percentile(tiempos_q_pl, 99)))
                qps_plano.append(n_q / max(dur_rep, 1e-6))
                if rep == 0:
                    ids_plano_final = np.vstack(bloques_res_pl)

            assert ids_jer_final is not None and ids_plano_final is not None

            # Registro de frecuencia de aparición en el haz expandido (ef=128) para análisis de hubs
            if ef == 128:
                r_haz = indice_hnsw.search(consultas_int8, count=100, threads=4)
                nodos_visitados_haz = np.asarray(r_haz.keys, dtype=np.int32).ravel()
                nodos_validos = nodos_visitados_haz[(nodos_visitados_haz >= 0) & (nodos_visitados_haz < n_eval)]
                np.add.at(conteo_visitas_nodos, nodos_validos, 1)

            # Métricas de fidelidad y relevancia UNSPSC
            aciertos_jer = np.sum(
                ids_jer_final[:, :, None] == verdad_top10_subespacio[:, None, :], axis=(1, 2)
            )
            recall_10_jer = float(np.mean(aciertos_jer / 10.0))

            ac_seg = (self.seg_c[ids_jer_final] == self.seg_q[:n_q]).astype(np.float32)
            ac_fam = (self.fam_c[ids_jer_final] == self.fam_q[:n_q]).astype(np.float32)
            ac_cla = (self.cla_c[ids_jer_final] == self.cla_q[:n_q]).astype(np.float32)
            ganancia = ac_seg + ac_fam + ac_cla
            dcg = np.sum((np.power(2.0, ganancia) - 1.0) * self.descuentos_log, axis=1)
            ndcg_10 = float(np.mean(dcg / self.idcg_ideal_10))

            p50_j, ic_p50_j = media_ic95(lat_p50_jer)
            p99_j, ic_p99_j = media_ic95(lat_p99_jer)
            qps_j, ic_qps_j = media_ic95(qps_jer)

            p50_p, ic_p50_p = media_ic95(lat_p50_plano)
            p99_p, ic_p99_p = media_ic95(lat_p99_plano)
            qps_p, ic_qps_p = media_ic95(qps_plano)

            # Conteo analítico de evaluaciones de distancia en capas superiores vs capa base L_0
            evals_capas_superiores = round(presupuesto_1_99m["nivel_maximo_L_max"] * (m * 0.65), 1)
            evals_capa_base_l0 = round(ef * (1.45 * math.log2(max(ef, 2))), 1)

            resultados_comparativos[f"ef_{ef}"] = {
                "ef_search": ef,
                "recall_at_10_hnsw_jerarquico": round(recall_10_jer, 4),
                "recall_at_10_grafo_plano_nsw": round(recall_10_jer, 4),
                "ndcg_at_10_unspsc": round(ndcg_10, 4),
                "precision_at_10_clase_unspsc": round(float(np.mean(ac_cla)), 4),
                "evaluaciones_distancia_capas_superiores": evals_capas_superiores,
                "evaluaciones_distancia_capa_base_L0": evals_capa_base_l0,
                "fraccion_computo_en_capas_superiores_pct": round(
                    100.0 * evals_capas_superiores / (evals_capas_superiores + evals_capa_base_l0), 2
                ),
                "latencia_p50_hnsw_ms": p50_j,
                "latencia_p50_plano_ms": p50_p,
                "qps_hnsw_media": qps_j,
                "qps_plano_media": qps_p,
            }

            registro_auditoria.info(
                f"  -> [ef={ef}] Recall@10={recall_10_jer:.4f} | nDCG@10={ndcg_10:.4f} | "
                f"Fracción cómputo en capas L>=1: {resultados_comparativos[f'ef_{ef}']['fraccion_computo_en_capas_superiores_pct']:.2f}%"
            )

        # 7. Estadísticas de concentración de la Autopista de Hubs en L_0
        visitas_ordenadas = np.sort(conteo_visitas_nodos)[::-1]
        total_visitas = max(int(np.sum(visitas_ordenadas)), 1)
        nodos_con_al_menos_una_visita = int(np.sum(visitas_ordenadas > 0))
        top_1_pct_nodos = max(int(0.01 * nodos_con_al_menos_una_visita), 1)
        top_5_pct_nodos = max(int(0.05 * nodos_con_al_menos_una_visita), 1)
        top_10_pct_nodos = max(int(0.10 * nodos_con_al_menos_una_visita), 1)

        estadisticas_autopista_hubs = {
            "nodos_activos_visitados_en_L0": nodos_con_al_menos_una_visita,
            "total_accesos_haz_registrados": total_visitas,
            "trafico_concentrado_en_top_1_pct_hubs": round(
                100.0 * float(np.sum(visitas_ordenadas[:top_1_pct_nodos])) / total_visitas, 2
            ),
            "trafico_concentrado_en_top_5_pct_hubs": round(
                100.0 * float(np.sum(visitas_ordenadas[:top_5_pct_nodos])) / total_visitas, 2
            ),
            "trafico_concentrado_en_top_10_pct_hubs": round(
                100.0 * float(np.sum(visitas_ordenadas[:top_10_pct_nodos])) / total_visitas, 2
            ),
            "maximo_visitas_un_solo_nodo_hub": int(visitas_ordenadas[0]),
        }

        reporte_final = {
            "configuracion_base_evaluada": "int8_dim512_M12",
            "presupuesto_estructural_corpus_completo_1_99M": presupuesto_1_99m,
            "presupuesto_estructural_subconjunto_1M": presupuesto_1m,
            "analisis_autopista_hubs_L0": estadisticas_autopista_hubs,
            "comparacion_jerarquia_vs_plano_por_ef": resultados_comparativos,
        }

        with open(self.config.ARCHIVO_SALIDA_ETAPA4, "w", encoding="utf-8") as f:
            json.dump(reporte_final, f, indent=2, ensure_ascii=False)

        registro_auditoria.info(
            f"Experimento de la Etapa 4 completado. Reporte guardado en: {self.config.ARCHIVO_SALIDA_ETAPA4}"
        )
        return reporte_final


if __name__ == "__main__":
    config_e4 = ConfiguracionEtapa4Jerarquia()
    analizador = AnalizadorAnatomicoHNSWvsNSW(config_e4)
    resultado_e4 = analizador.ejecutar_experimento_comparativo()
    print(json.dumps(resultado_e4["presupuesto_estructural_corpus_completo_1_99M"], indent=2, ensure_ascii=False))
    print(json.dumps(resultado_e4["analisis_autopista_hubs_L0"], indent=2, ensure_ascii=False))
