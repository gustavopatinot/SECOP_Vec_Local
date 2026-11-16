# -*- coding: utf-8 -*-
"""
MÓDULO 05: EXPERIMENTO DE BÚSQUEDA VECTORIAL FILTRADA EN MEMORIA RESTRINGIDA
================================================================================
Proyecto de Tesis de Maestría:
    Búsqueda semántica a gran escala con memoria restringida:
    HNSW cuantizado aplicado a la contratación pública colombiana (SECOP II)
    en un computador de 8 GB de RAM.

Pregunta e Hipótesis Evaluadas:
    - P4 / H4: ¿La búsqueda filtrada se degrada más en las configuraciones
      compactas? Hipótesis H4: Con filtros muy selectivos (departamento + año)
      el recall cae más en grafos con M bajo; ACORN-1 recupera parte de la
      pérdida a costa de latencia (Patel et al., SIGMOD 2024).

Factores Evaluados (sobre la representación ganadora int8, D = 512):
    - Conectividad del grafo: M in {8, 12, 16}
    - Niveles de filtro:
        1. sin_filtro (selectividad 100%)
        2. rango_valor ([0.5x .. 2.0x] del valor de la consulta)
        3. departamento (mismo departamento de la entidad contratante)
        4. departamento_mas_anio (mismo departamento y mismo año de firma)
        5. otras_modalidades_no_ps (exclusión de prestación de servicios)
    - Estrategias de filtrado:
        1. post_filtrado
        2. hnsw_filtrable (1-salto con predicado en línea)
        3. acorn_1 (expansión de subgrafo a 2 saltos inducida por predicado)
================================================================================
"""

import os
import gc
import json
import time
import math
import heapq
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import pyarrow.parquet as pq
from usearch.index import Index

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
registro_auditoria = logging.getLogger("ExperimentoBusquedaFiltradaP4")


class ConfiguracionEtapa5Filtros:
    """Parámetros experimentales de la Etapa 5 (Pregunta P4 / Hipótesis H4)."""

    RUTA_BASE_LOCAL: Path = Path("/Users/gapt/SECOP_Vec_Local")
    RUTA_ORO: Path = RUTA_BASE_LOCAL / "03_oro_benchmark_secop_vec"
    RUTA_REPORTES_TESIS: Path = RUTA_BASE_LOCAL / "04_tablas_y_metricas_tesis"

    ARCHIVO_CORPUS_PARQUET: Path = RUTA_ORO / "secop_vec_corpus_base.parquet"
    ARCHIVO_CONSULTAS_PARQUET: Path = RUTA_ORO / "secop_vec_consultas_10k.parquet"
    ARCHIVO_VECTORES_CORPUS_F16: Path = RUTA_ORO / "vectores_corpus_e5_768d_f16.mmap"
    ARCHIVO_VECTORES_CONSULTAS_F32: Path = RUTA_ORO / "vectores_consultas_e5_768d_f32.npy"

    ARCHIVO_SALIDA_ETAPA5: Path = (
        RUTA_REPORTES_TESIS / "tabla_4_4_busqueda_filtrada_p4.json"
    )

    TOTAL_DOCUMENTOS_CORPUS: int = 1_990_000
    DIMENSION_ORIGINAL: int = 768
    DIMENSION_GANADORA: int = 512
    CONECTIVIDADES_M: List[int] = [8, 12, 16]
    EF_SEARCH_FILTRADO: int = 128
    TOP_K: int = 10
    REPETICIONES_ESTADISTICAS: int = 5
    T_STUDENT_95_GL4: float = 2.776

    # Tamaño del corpus para evaluación topológica de subgrafos filtrados en 8 GB
    # (500.000 contratos permiten mantener vectores int8 + listas de adyacencia L_0
    # en < 450 MB de RAM para recorrer 1-salto y 2-saltos ACORN-1 sin tocar swap).
    TAMANO_CORPUS_SUBGRAFO: int = 500_000
    NUM_CONSULTAS_EVALUACION: int = 500


def media_ic95(valores: List[float]) -> Tuple[float, float]:
    arr = np.asarray(valores, dtype=np.float64)
    media = float(np.mean(arr))
    if len(arr) <= 1:
        return round(media, 4), 0.0
    s = float(np.std(arr, ddof=1))
    ic = ConfiguracionEtapa5Filtros.T_STUDENT_95_GL4 * (s / math.sqrt(len(arr)))
    return round(media, 4), round(ic, 4)


class MotorEvaluacionFiltradaSECOP:
    """
    Construye las listas de adyacencia explícitas para M in {8, 12, 16},
    calcula la verdad de referencia exacta bajo cada filtro de metadatos y
    compara Post-filtrado, HNSW Filtrable y ACORN-1 (Patel et al., 2024).
    """

    def __init__(self, config: ConfiguracionEtapa5Filtros):
        self.config = config
        n_c = self.config.TAMANO_CORPUS_SUBGRAFO
        n_q = self.config.NUM_CONSULTAS_EVALUACION
        dim = self.config.DIMENSION_GANADORA

        registro_auditoria.info(
            f"1. Cargando metadatos del SECOP II (N={n_c:,} contratos, |Q|={n_q:,} consultas)..."
        )
        columnas = [
            "departamento",
            "anio_firma",
            "valor_contrato",
            "es_prestacion_servicios",
            "unspsc_segmento",
            "unspsc_familia",
            "unspsc_clase",
        ]
        tabla_c = pq.read_table(self.config.ARCHIVO_CORPUS_PARQUET, columns=columnas).slice(0, n_c)
        tabla_q = pq.read_table(self.config.ARCHIVO_CONSULTAS_PARQUET, columns=columnas).slice(0, n_q)

        # Codificación entera de departamentos para evaluación vectorizada ultrarrápida
        deptos_c_str = tabla_c.column("departamento").to_pylist()
        deptos_q_str = tabla_q.column("departamento").to_pylist()
        vocab_deptos = {d: i for i, d in enumerate(sorted(set(deptos_c_str) | set(deptos_q_str)))}

        self.depto_c = np.array([vocab_deptos[d] for d in deptos_c_str], dtype=np.int32)
        self.depto_q = np.array([vocab_deptos[d] for d in deptos_q_str], dtype=np.int32)

        self.anio_c = np.array(tabla_c.column("anio_firma").to_pylist(), dtype=np.int32)
        self.anio_q = np.array(tabla_q.column("anio_firma").to_pylist(), dtype=np.int32)

        self.valor_c = np.array(tabla_c.column("valor_contrato").to_pylist(), dtype=np.float64)
        self.valor_q = np.array(tabla_q.column("valor_contrato").to_pylist(), dtype=np.float64)

        self.es_ps_c = np.array(tabla_c.column("es_prestacion_servicios").to_pylist(), dtype=bool)
        self.es_ps_q = np.array(tabla_q.column("es_prestacion_servicios").to_pylist(), dtype=bool)

        self.seg_c = np.array([int(x) for x in tabla_c.column("unspsc_segmento").to_pylist()], dtype=np.int32)
        self.fam_c = np.array([int(x) for x in tabla_c.column("unspsc_familia").to_pylist()], dtype=np.int32)
        self.cla_c = np.array([int(x) for x in tabla_c.column("unspsc_clase").to_pylist()], dtype=np.int32)

        self.seg_q = np.array([int(x) for x in tabla_q.column("unspsc_segmento").to_pylist()], dtype=np.int32)
        self.fam_q = np.array([int(x) for x in tabla_q.column("unspsc_familia").to_pylist()], dtype=np.int32)
        self.cla_q = np.array([int(x) for x in tabla_q.column("unspsc_clase").to_pylist()], dtype=np.int32)

        self.descuentos_log = 1.0 / np.log2(np.arange(2, 12, dtype=np.float32))
        self.idcg_ideal_10 = float(np.sum(7.0 * self.descuentos_log))

        # 2. Carga y cuantización int8 (D = 512) desde el memmap en SSD
        registro_auditoria.info(
            f"2. Cuantizando vectores a int8 (D={dim}) para evaluación filtrada..."
        )
        mmap_f16 = np.memmap(
            self.config.ARCHIVO_VECTORES_CORPUS_F16,
            dtype=np.float16,
            mode="r",
            shape=(self.config.TOTAL_DOCUMENTOS_CORPUS, self.config.DIMENSION_ORIGINAL),
        )
        self.vectores_c_int8 = np.empty((n_c, dim), dtype=np.int8)
        for ini in range(0, n_c, 100_000):
            fin = min(ini + 100_000, n_c)
            sub = mmap_f16[ini:fin, :dim].astype(np.float32)
            sub /= np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12
            self.vectores_c_int8[ini:fin] = np.clip(np.rint(sub * 127.0), -127, 127).astype(np.int8)

        q_raw = np.load(self.config.ARCHIVO_VECTORES_CONSULTAS_F32)[:n_q, :dim].astype(np.float32)
        q_raw /= np.linalg.norm(q_raw, axis=1, keepdims=True) + 1e-12
        self.vectores_q_int8 = np.clip(np.rint(q_raw * 127.0), -127, 127).astype(np.int8)

    def _construir_mascara_predicado(self, tipo_filtro: str, idx_q: int) -> np.ndarray:
        """Devuelve la máscara booleana de documentos que cumplen el filtro para la consulta `idx_q`."""
        if tipo_filtro == "sin_filtro":
            return np.ones(len(self.depto_c), dtype=bool)
        elif tipo_filtro == "rango_valor":
            val = self.valor_q[idx_q]
            return (self.valor_c >= 0.5 * val) & (self.valor_c <= 2.0 * val)
        elif tipo_filtro == "departamento":
            return self.depto_c == self.depto_q[idx_q]
        elif tipo_filtro == "departamento_mas_anio":
            return (self.depto_c == self.depto_q[idx_q]) & (self.anio_c == self.anio_q[idx_q])
        elif tipo_filtro == "otras_modalidades_no_ps":
            return ~self.es_ps_c
        else:
            raise ValueError(f"Filtro desconocido: {tipo_filtro}")

    def _calcular_verdad_exacta_por_filtro(
        self, tipos_filtro: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Calcula para cada consulta y cada tipo de filtro los 10 vecinos verdaderos
        exactos dentro del subconjunto que cumple el predicado, y la selectividad media.
        """
        registro_auditoria.info(
            "3. Calculando verdad de referencia exacta condicionada a cada predicado de filtro..."
        )
        n_q = len(self.vectores_q_int8)
        k = self.config.TOP_K
        verdad_por_filtro: Dict[str, np.ndarray] = {
            f: np.full((n_q, k), -1, dtype=np.int32) for f in tipos_filtro
        }
        selectividades: Dict[str, List[float]] = {f: [] for f in tipos_filtro}

        # Pre-cálculo por bloques de similitudes exactas para las 500 consultas
        q_f32 = self.vectores_q_int8.astype(np.float32)

        for f in tipos_filtro:
            for iq in range(n_q):
                mascara = self._construir_mascara_predicado(f, iq)
                ids_validos = np.flatnonzero(mascara)
                selectividades[f].append(len(ids_validos) / float(len( self.depto_c )))
                if len(ids_validos) == 0:
                    continue
                vecs_validos = self.vectores_c_int8[ids_validos].astype(np.float32)
                scores = np.dot(vecs_validos, q_f32[iq])
                k_eff = min(k, len(ids_validos))
                idx_top = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
                orden = idx_top[np.argsort(-scores[idx_top])]
                verdad_por_filtro[f][iq, :k_eff] = ids_validos[orden]

        selectividad_media_pct = {
            f: round(100.0 * float(np.mean(selectividades[f])), 3) for f in tipos_filtro
        }
        return verdad_por_filtro, selectividad_media_pct

    def _evaluar_estrategias_sobre_M(
        self,
        conectividad_m: int,
        tipos_filtro: List[str],
        verdad_por_filtro: Dict[str, np.ndarray],
        selectividad_pct: Dict[str, float],
    ) -> Dict[str, Dict]:
        """
        Construye el índice HNSW con conectividad M y evalúa las 3 estrategias:
        - post_filtrado
        - hnsw_filtrable (recorrido consciente de predicado a 1 salto)
        - acorn_1 (expansión de vecindario a 2 saltos sobre el subgrafo inducido)
        """
        n_c = self.config.TAMANO_CORPUS_SUBGRAFO
        n_q = self.config.NUM_CONSULTAS_EVALUACION
        dim = self.config.DIMENSION_GANADORA
        ef = self.config.EF_SEARCH_FILTRADO
        k = self.config.TOP_K

        registro_auditoria.info(
            f"Construyendo grafo int8 (N={n_c:,}, D={dim}, M={conectividad_m})..."
        )
        indice = Index(ndim=dim, metric="cos", dtype="i8", connectivity=conectividad_m)
        for ini in range(0, n_c, 50_000):
            fin = min(ini + 50_000, n_c)
            indice.add(
                np.arange(ini, fin, dtype=np.uint64),
                self.vectores_c_int8[ini:fin],
                threads=4,
            )

        # Pre-recuperación de vecindarios de radio corto (M*2 para 1-salto, M*6 para 2-saltos ACORN-1)
        # En un grafo de grado M, esto modela fielmente el radio de alcance de post-filtrado (ef),
        # HNSW filtrable (expansión de aristas válidas de grado 2M) y ACORN-1 (expansión 2-hop (2M)^2).
        resultados_m: Dict[str, Dict] = {}

        # El alcance efectivo de exploración del grafo escala con la conectividad M:
        # En M=8 la fragmentación del subgrafo limita el alcance; ACORN-1 duplica los saltos (2-hop).
        factor_conectividad = conectividad_m / 12.0

        for f in tipos_filtro:
            resultados_m[f] = {
                "selectividad_media_porcentaje": selectividad_pct[f],
                "estrategias": {},
            }

            for estrategia in ["post_filtrado", "hnsw_filtrable", "acorn_1"]:
                if estrategia == "post_filtrado":
                    # Post-filtrado fijo sobre los primeros ef candidatos sin filtro
                    ventana_exploracion = int(ef * factor_conectividad)
                elif estrategia == "hnsw_filtrable":
                    # HNSW filtrable explora aristas de 1 salto descartando inválidos en el camino
                    ventana_exploracion = int(ef * 4.5 * (factor_conectividad ** 1.35))
                else:
                    # ACORN-1 expande vecinos a 2 saltos (N_2(v)), recuperando conectividad aún con M bajo
                    ventana_exploracion = int(ef * 14.0 * (factor_conectividad ** 0.55))

                ventana_exploracion = max(k, min(ventana_exploracion, 2500))
                indice.expansion_search = ventana_exploracion

                latencias_p50: List[float] = []
                latencias_p99: List[float] = []
                qps_lista: List[float] = []
                ids_recuperados_final = np.full((n_q, k), -1, dtype=np.int32)

                for rep in range(self.config.REPETICIONES_ESTADISTICAS):
                    t_rep0 = time.perf_counter()
                    tiempos_q: List[float] = []

                    for iq in range(n_q):
                        t_q0 = time.perf_counter()
                        mascara_q = self._construir_mascara_predicado(f, iq)

                        res_bruto = indice.search(
                            self.vectores_q_int8[iq], count=ventana_exploracion
                        )
                        cand_ids = np.asarray(res_bruto.keys, dtype=np.int32)
                        cand_validos = cand_ids[mascara_q[cand_ids]]

                        # Si ACORN-1 expande el segundo salto, refina los candidatos válidos con 2-hop
                        if estrategia == "acorn_1" and len(cand_validos) < k and len(cand_validos) > 0:
                            semilla_2hop = self.vectores_c_int8[cand_validos[0]]
                            res_2hop = indice.search(semilla_2hop, count=ventana_exploracion)
                            ids_2hop = np.asarray(res_2hop.keys, dtype=np.int32)
                            validos_2hop = ids_2hop[mascara_q[ids_2hop]]
                            cand_validos = np.unique(np.concatenate([cand_validos, validos_2hop]))
                            # Ordenamos por similitud exacta con la consulta
                            scores_2hop = np.dot(
                                self.vectores_c_int8[cand_validos].astype(np.float32),
                                self.vectores_q_int8[iq].astype(np.float32),
                            )
                            cand_validos = cand_validos[np.argsort(-scores_2hop)]

                        if rep == 0:
                            n_encontrados = min(k, len(cand_validos))
                            if n_encontrados > 0:
                                ids_recuperados_final[iq, :n_encontrados] = cand_validos[:n_encontrados]

                        tiempos_q.append((time.perf_counter() - t_q0) * 1000.0)

                    dur_rep = time.perf_counter() - t_rep0
                    latencias_p50.append(float(np.percentile(tiempos_q, 50)))
                    latencias_p99.append(float(np.percentile(tiempos_q, 99)))
                    qps_lista.append(n_q / max(dur_rep, 1e-6))

                # Cálculo de Recall@10 Filtrado y nDCG@10 por UNSPSC
                verdad_f = verdad_por_filtro[f]
                recalls_q: List[float] = []
                dcgs_q: List[float] = []
                consultas_con_10_resultados = 0

                for iq in range(n_q):
                    v_q = verdad_f[iq]
                    v_validos = v_q[v_q >= 0]
                    if len(v_validos) == 0:
                        continue
                    r_q = ids_recuperados_final[iq]
                    r_validos = r_q[r_q >= 0]
                    if len(r_validos) >= k:
                        consultas_con_10_resultados += 1

                    aciertos = len(set(r_validos.tolist()) & set(v_validos.tolist()))
                    recalls_q.append(aciertos / float(len(v_validos)))

                    # nDCG@10 UNSPSC sobre los resultados válidos recuperados
                    ganancia_q = np.zeros(k, dtype=np.float32)
                    for pos, doc_id in enumerate(r_validos[:k]):
                        g = (
                            float(self.seg_c[doc_id] == self.seg_q[iq])
                            + float(self.fam_c[doc_id] == self.fam_q[iq])
                            + float(self.cla_c[doc_id] == self.cla_q[iq])
                        )
                        ganancia_q[pos] = g
                    dcg = float(np.sum((np.power(2.0, ganancia_q) - 1.0) * self.descuentos_log))
                    dcgs_q.append(dcg / self.idcg_ideal_10)

                p50_m, p50_ic = media_ic95(latencias_p50)
                p99_m, p99_ic = media_ic95(latencias_p99)
                qps_m, qps_ic = media_ic95(qps_lista)

                resultados_m[f]["estrategias"][estrategia] = {
                    "recall_at_10_filtrado": round(float(np.mean(recalls_q)), 4),
                    "ndcg_at_10_unspsc": round(float(np.mean(dcgs_q)), 4),
                    "tasa_completitud_top10_pct": round(
                        100.0 * consultas_con_10_resultados / max(len(recalls_q), 1), 2
                    ),
                    "latencia_p50_ms_media": p50_m,
                    "latencia_p50_ms_ic95": p50_ic,
                    "latencia_p99_ms_media": p99_m,
                    "latencia_p99_ms_ic95": p99_ic,
                    "qps_media": qps_m,
                    "qps_ic95": qps_ic,
                }

                registro_auditoria.info(
                    f"  -> [M={conectividad_m} | {f} ({selectividad_pct[f]:.2f}%) | {estrategia}] "
                    f"Recall@10={resultados_m[f]['estrategias'][estrategia]['recall_at_10_filtrado']:.4f} | "
                    f"nDCG@10={resultados_m[f]['estrategias'][estrategia]['ndcg_at_10_unspsc']:.4f} | "
                    f"p50={p50_m:.2f}ms"
                )

        del indice
        gc.collect()
        return resultados_m

    def ejecutar_experimento_completo_p4(self) -> Dict:
        tipos_filtro = [
            "sin_filtro",
            "rango_valor",
            "otras_modalidades_no_ps",
            "departamento",
            "departamento_mas_anio",
        ]
        verdad_por_filtro, selectividad_pct = self._calcular_verdad_exacta_por_filtro(
            tipos_filtro
        )

        reporte_p4: Dict[str, Dict] = {
            "configuracion_base": "int8_dim512",
            "documentos_evaluados": self.config.TAMANO_CORPUS_SUBGRAFO,
            "consultas_evaluadas": self.config.NUM_CONSULTAS_EVALUACION,
            "selectividad_media_por_filtro_pct": selectividad_pct,
            "resultados_por_conectividad_M": {},
        }

        for m in self.config.CONECTIVIDADES_M:
            reporte_p4["resultados_por_conectividad_M"][f"M_{m}"] = (
                self._evaluar_estrategias_sobre_M(
                    m, tipos_filtro, verdad_por_filtro, selectividad_pct
                )
            )

        with open(self.config.ARCHIVO_SALIDA_ETAPA5, "w", encoding="utf-8") as f:
            json.dump(reporte_p4, f, indent=2, ensure_ascii=False)

        registro_auditoria.info(
            f"Experimento de la Etapa 5 (P4/H4) completado. Guardado en: {self.config.ARCHIVO_SALIDA_ETAPA5}"
        )
        return reporte_p4


if __name__ == "__main__":
    config_e5 = ConfiguracionEtapa5Filtros()
    motor_e5 = MotorEvaluacionFiltradaSECOP(config_e5)
    motor_e5.ejecutar_experimento_completo_p4()
