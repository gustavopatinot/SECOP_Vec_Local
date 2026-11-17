# -*- coding: utf-8 -*-
"""
MÓDULO 12: INTERFAZ WEB INTERACTIVA DEL BUSCADOR SEMÁNTICO Y CAPA RAG (SECOP-VEC)
================================================================================
Proyecto de Investigación Aplicada — Universidad Santo Tomás (MADSI)
Autor: Gustavo Adolfo Patiño Toro

Incluye gestor de descarga explícita con barras de progreso en consola para
verificar artefactos locales y descargar los modelos desde Hugging Face Hub.
================================================================================
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import duckdb
import gradio as gr
import numpy as np
import psutil
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from usearch.index import Index

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
registro = logging.getLogger("InterfazWebSECOPVec")

RUTA_BASE = Path("/Users/gapt/SECOP_Vec_Local")
RUTA_ORO = RUTA_BASE / "03_oro_benchmark_secop_vec"
RUTA_INDICES = RUTA_BASE / "05_indices_temporales_ssd"

ARCHIVO_CORPUS_PARQUET = RUTA_ORO / "secop_vec_corpus_base.parquet"
ARCHIVO_INDICE_USEARCH = RUTA_INDICES / "secop_int8_dim512_M12.usearch"

NOMBRE_MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"
NOMBRE_MODELO_LLM_MLX = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DIMENSION_MATRYOSHKA = 512


def obtener_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def verificar_y_descargar_componentes(incluir_llm_mlx: bool = True) -> None:
    """
    Audita los artefactos locales en disco y descarga explícitamente los modelos
    neuronales mostrando barras de progreso detalladas en la terminal.
    """
    print("\n" + "=" * 78)
    print(" AUDITORÍA Y DESCARGA VISIBLE DE COMPONENTES DEL SISTEMA SECOP-VEC (8 GB RAM)")
    print("=" * 78)

    # [1/4] Verificación del índice vectorial HNSW cuantizado (.usearch)
    print("\n[1/4] Verificando índice vectorial HNSW cuantizado (int8, 512D, M=12)...")
    if not ARCHIVO_INDICE_USEARCH.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontró {ARCHIVO_INDICE_USEARCH}. "
            "Ejecuta primero `make etapa6` o descárgalo desde Google Drive."
        )
    peso_indice_mb = ARCHIVO_INDICE_USEARCH.stat().st_size / (1024 ** 2)
    print(f"      -> OK: {ARCHIVO_INDICE_USEARCH.name} ({peso_indice_mb:,.2f} MB listos para view=True)")

    # [2/4] Verificación del corpus columnar Oro (.parquet)
    print("\n[2/4] Verificando corpus columnar Oro (1.990.000 contratos únicos)...")
    if not ARCHIVO_CORPUS_PARQUET.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontró {ARCHIVO_CORPUS_PARQUET}. "
            "Verifica la carpeta 03_oro_benchmark_secop_vec."
        )
    peso_parquet_mb = ARCHIVO_CORPUS_PARQUET.stat().st_size / (1024 ** 2)
    print(f"      -> OK: {ARCHIVO_CORPUS_PARQUET.name} ({peso_parquet_mb:,.2f} MB en SSD)")

    # [3/4] Descarga explícita con barra de progreso del modelo de embeddings E5
    print(f"\n[3/4] Descargando / Verificando codificador semántico '{NOMBRE_MODELO_EMBEDDINGS}'...")
    ruta_e5 = snapshot_download(
        repo_id=NOMBRE_MODELO_EMBEDDINGS,
        tqdm_class=tqdm,
    )
    print(f"      -> OK: Codificador E5 verificado en caché local: {ruta_e5}")

    # [4/4] Descarga explícita con barra de progreso del LLM generativo 4-bit (MLX)
    if incluir_llm_mlx:
        print(f"\n[4/4] Descargando / Verificando LLM Forense 4-bit '{NOMBRE_MODELO_LLM_MLX}'...")
        try:
            ruta_llm = snapshot_download(
                repo_id=NOMBRE_MODELO_LLM_MLX,
                tqdm_class=tqdm,
            )
            print(f"      -> OK: LLM Generativo MLX 4-bit verificado en caché local: {ruta_llm}")
        except Exception as err:
            print(f"      -> [AVISO] No se pudo pre-descargar el LLM opcional ({err}).")
    else:
        print("\n[4/4] Pre-descarga de LLM MLX omitida por parámetro (--sin-pre-descarga-llm).")

    print("\n" + "=" * 78)
    print(" TODOS LOS COMPONENTES ESTÁN DESCARGADOS Y VERIFICADOS. INICIANDO SERVIDOR...")
    print("=" * 78 + "\n")
    sys.stdout.flush()


class MotorWebSECOPVecRAG:
    """Motor híbrido de búsqueda semántica y síntesis RAG para equipos de 8 GB."""

    def __init__(self) -> None:
        registro.info("1. Mapeando índice secop_int8_dim512_M12.usearch desde SSD (view=True)...")
        self.indice = Index.restore(str(ARCHIVO_INDICE_USEARCH), view=True)

        registro.info("2. Conectando DuckDB fuera de núcleo sobre secop_vec_corpus_base.parquet...")
        self.db = duckdb.connect(database=":memory:")
        self.db.execute("PRAGMA threads=4;")
        self.db.execute(f"""
            CREATE VIEW contratos AS
            SELECT
                (row_number() OVER () - 1)::BIGINT AS idx_vector,
                id_contrato,
                numero_proceso,
                descripcion_objeto,
                unspsc_8digitos,
                unspsc_segmento,
                unspsc_clase,
                departamento,
                entidad,
                modalidad,
                tipo_contrato,
                es_prestacion_servicios,
                valor_contrato,
                anio_firma
            FROM read_parquet('{ARCHIVO_CORPUS_PARQUET}')
        """)

        self.departamentos_disponibles = ["TODOS"] + [
            fila[0]
            for fila in self.db.execute(
                "SELECT DISTINCT departamento FROM contratos WHERE departamento IS NOT NULL ORDER BY departamento"
            ).fetchall()
        ]
        self.anios_disponibles = ["TODOS"] + [
            str(fila[0])
            for fila in self.db.execute(
                "SELECT DISTINCT anio_firma FROM contratos WHERE anio_firma IS NOT NULL ORDER BY anio_firma DESC"
            ).fetchall()
        ]

        registro.info(f"3. Inicializando tensores de '{NOMBRE_MODELO_EMBEDDINGS}' en memoria unificada...")
        dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
        self.codificador = SentenceTransformer(NOMBRE_MODELO_EMBEDDINGS, device=dispositivo)
        self.codificador.max_seq_length = 160
        if dispositivo == "mps":
            self.codificador.half()

        self.modelo_llm = None
        self.tokenizador_llm = None
        registro.info(f"Motor listo en {dispositivo.upper()} | Memoria RSS base: {obtener_rss_mb():.2f} MB")

    def _vectorizar_pregunta_matryoshka(self, pregunta: str) -> np.ndarray:
        vec_768 = self.codificador.encode(
            [f"query: {pregunta.strip()}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        vec_512 = vec_768[:DIMENSION_MATRYOSHKA].copy()
        vec_512 /= np.linalg.norm(vec_512) + 1e-12
        return vec_512

    def _cargar_llm_mlx_bajo_demanda(self) -> bool:
        if self.modelo_llm is not None:
            return True
        try:
            from mlx_lm import load  # type: ignore
            registro.info(f"Montando tensores 4-bit en GPU Metal: {NOMBRE_MODELO_LLM_MLX}...")
            self.modelo_llm, self.tokenizador_llm = load(NOMBRE_MODELO_LLM_MLX)
            return True
        except Exception as err:
            registro.warning(f"No fue posible inicializar MLX LLM ({err}). Operando con síntesis determinista.")
            return False

    def buscar_y_sintetizar(
        self,
        consulta_texto: str,
        departamento: str,
        anio: str,
        valor_min_millones: float,
        valor_max_millones: float,
        excluir_prestacion_servicios: bool,
        top_k: int,
        usar_llm_generativo: bool,
        progreso: gr.Progress = gr.Progress(),
    ) -> Tuple[str, str, List[List]]:
        if not consulta_texto or not consulta_texto.strip():
            return "Por favor ingresa una descripción contractual para buscar.", "", []

        t_inicio_total = time.perf_counter()
        valor_min_millones = float(valor_min_millones or 0.0)
        valor_max_millones = float(valor_max_millones or 0.0)
        departamento = departamento or "TODOS"
        anio = anio or "TODOS"
        top_k = int(top_k or 10)

        progreso(0.15, desc="Vectorizando consulta (E5-base -> Matryoshka 512D)...")
        t0 = time.perf_counter()
        vector_q = self._vectorizar_pregunta_matryoshka(consulta_texto)
        t_emb_ms = (time.perf_counter() - t0) * 1000.0

        progreso(0.40, desc="Explorando grafo HNSW/ACORN-1 mapeado en SSD (view=True)...")
        t1 = time.perf_counter()
        tiene_filtro_selectivo = (
            departamento != "TODOS"
            or anio != "TODOS"
            or excluir_prestacion_servicios
            or valor_min_millones > 0
            or valor_max_millones > 0
        )
        ventana_busqueda = 2500 if tiene_filtro_selectivo else max(128, int(top_k) * 5)
        estrategia_aplicada = (
            "ACORN-1 Adaptativo (Expansión 2-saltos)"
            if tiene_filtro_selectivo
            else "HNSW / Plano NSW Directo (ef=128)"
        )

        self.indice.expansion_search = ventana_busqueda
        res_ann = self.indice.search(vector_q, count=ventana_busqueda)
        ids_candidatos = np.asarray(res_ann.keys, dtype=np.int64)
        distancias = np.asarray(res_ann.distances, dtype=np.float32)
        t_ann_ms = (time.perf_counter() - t1) * 1000.0

        progreso(0.65, desc="Filtrando metadatos y agregando estadísticas con DuckDB...")
        t2 = time.perf_counter()
        mapa_similitud = {
            int(idx): round(max(0.0, 1.0 - float(dist)), 4)
            for idx, dist in zip(ids_candidatos, distancias)
        }

        condiciones_sql = [f"idx_vector IN ({','.join(map(str, ids_candidatos.tolist()))})"]
        if departamento != "TODOS":
            depto_limpio = departamento.replace("'", "''")
            condiciones_sql.append(f"departamento = '{depto_limpio}'")
        if anio != "TODOS":
            condiciones_sql.append(f"anio_firma = {int(anio)}")
        if valor_min_millones > 0:
            condiciones_sql.append(f"valor_contrato >= {valor_min_millones * 1_000_000.0}")
        if valor_max_millones > 0:
            condiciones_sql.append(f"valor_contrato <= {valor_max_millones * 1_000_000.0}")
        if excluir_prestacion_servicios:
            condiciones_sql.append("es_prestacion_servicios = FALSE AND unspsc_segmento != '80'")

        clausula_where = " AND ".join(condiciones_sql)
        filas_sql = self.db.execute(f"""
            SELECT
                idx_vector,
                numero_proceso,
                departamento,
                anio_firma,
                entidad,
                modalidad,
                unspsc_clase,
                valor_contrato,
                descripcion_objeto
            FROM contratos
            WHERE {clausula_where}
        """).fetchall()

        filas_ordenadas = sorted(
            filas_sql, key=lambda r: mapa_similitud.get(int(r[0]), 0.0), reverse=True
        )[: int(top_k)]
        t_sql_ms = (time.perf_counter() - t2) * 1000.0

        if not filas_ordenadas:
            rss_actual = obtener_rss_mb()
            telemetria = (
                f"**Estrategia:** {estrategia_aplicada} | **RAM RSS:** {rss_actual:.1f} MB / 8.192 MB"
            )
            return (
                "No se encontraron contratos con la combinación exacta de filtros aplicados.",
                telemetria,
                [],
            )

        valores = np.array([float(r[7]) for r in filas_ordenadas], dtype=np.float64)
        val_min, val_med, val_prom, val_max = (
            float(np.min(valores)),
            float(np.median(valores)),
            float(np.mean(valores)),
            float(np.max(valores)),
        )

        conteo_entidades = {}
        conteo_modalidades = {}
        conteo_clases = {}
        tabla_salida: List[List] = []

        for rank, r in enumerate(filas_ordenadas, start=1):
            idx_v, num_proc, depto, anio_f, ent, mod, clase_u, val_cop, desc = r
            sim = mapa_similitud.get(int(idx_v), 0.0)
            conteo_entidades[ent] = conteo_entidades.get(ent, 0) + 1
            conteo_modalidades[mod] = conteo_modalidades.get(mod, 0) + 1
            conteo_clases[clase_u] = conteo_clases.get(clase_u, 0) + 1

            tabla_salida.append([
                rank,
                f"{sim:.4f}",
                num_proc,
                f"{depto} ({anio_f})",
                ent[:45],
                mod[:32],
                clase_u,
                f"${val_cop:,.0f}",
                desc[:260],
            ])

        top_ent_str = ", ".join(
            [f"**{k}** ({v})" for k, v in sorted(conteo_entidades.items(), key=lambda x: -x[1])[:3]]
        )
        top_mod_str = ", ".join(
            [f"**{k}** ({v})" for k, v in sorted(conteo_modalidades.items(), key=lambda x: -x[1])[:3]]
        )
        top_cls_str = ", ".join(
            [f"`{k}` ({v})" for k, v in sorted(conteo_clases.items(), key=lambda x: -x[1])[:3]]
        )

        resumen_rag = (
            f"### Síntesis Analítica RAG ({len(filas_ordenadas)} contratos análogos recuperados)\n\n"
            f"- **Rango Presupuestal (COP):** Mínimo **${val_min:,.0f}** | Mediana **${val_med:,.0f}** | "
            f"Promedio **${val_prom:,.0f}** | Máximo **${val_max:,.0f}**.\n"
            f"- **Modalidades de Selección Predominantes:** {top_mod_str}.\n"
            f"- **Entidades Contratantes más Recurrentes:** {top_ent_str}.\n"
            f"- **Clases UNSPSC Identificadas:** {top_cls_str}.\n"
        )

        if usar_llm_generativo:
            progreso(0.85, desc="Generando síntesis con Qwen2.5-3B-Instruct-4bit en Metal GPU...")
            if self._cargar_llm_mlx_bajo_demanda():
                from mlx_lm import generate  # type: ignore
                t_llm0 = time.perf_counter()
                contexto_contratos = "\n".join(
                    [f"- {r[4]} (${r[7]:,.0f} COP, {r[5]}): {r[8][:180]}" for r in filas_ordenadas[:5]]
                )
                prompt_rag = (
                    f"<|im_start|>system\nEres un auditor experto en contratación pública colombiana (SECOP II). "
                    f"Redacta un diagnóstico conciso en 4 líneas basado exclusivamente en los contratos recuperados.<|im_end|>\n"
                    f"<|im_start|>user\nConsulta: {consulta_texto}\nMediana: ${val_med:,.0f} COP.\n"
                    f"Contratos:\n{contexto_contratos}<|im_end|>\n<|im_start|>assistant\n"
                )
                texto_llm = generate(
                    self.modelo_llm,
                    self.tokenizador_llm,
                    prompt=prompt_rag,
                    max_tokens=180,
                    verbose=False,
                )
                t_llm_ms = (time.perf_counter() - t_llm0) * 1000.0
                resumen_rag += (
                    f"\n> **Diagnóstico Generativo Local (Qwen2.5-3B-4bit en Metal — {t_llm_ms:.0f} ms):** "
                    f"{texto_llm.strip()}\n"
                )

        progreso(1.0, desc="Consulta completada.")
        t_total_ms = (time.perf_counter() - t_inicio_total) * 1000.0
        rss_actual = obtener_rss_mb()
        margen_libre_mb = 8192.0 - rss_actual

        telemetria_md = (
            f"**Estrategia:** {estrategia_aplicada} | "
            f"**Tiempos:** Embedding `{t_emb_ms:.1f} ms` + Búsqueda Vectorial `{t_ann_ms:.1f} ms` + "
            f"Filtro/RAG `{t_sql_ms:.1f} ms` (**Total: `{t_total_ms:.1f} ms`**) | "
            f"**Memoria Física RSS:** `{rss_actual:.1f} MB` (Libres en 8 GB: `{margen_libre_mb:.1f} MB`)"
        )

        return resumen_rag, telemetria_md, tabla_salida


def construir_interfaz_gradio(motor: MotorWebSECOPVecRAG) -> gr.Blocks:
    with gr.Blocks(title="SECOP-Vec | Buscador Semántico y RAG en 8 GB RAM", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# SECOP-Vec: Búsqueda Semántica y Auditoría RAG sobre 1.990.000 Contratos\n"
            "**Tesis de Maestría en Análisis de Datos y Sistemas Inteligentes — Universidad Santo Tomás** | "
            "Índice cuantizado `int8` (512D Matryoshka, $M=12$, `1.193,23 MB` mapeado en SSD con `view=True`) "
            "ejecutándose en hardware de **8 GB de RAM**."
        )

        with gr.Row():
            consulta_input = gr.Textbox(
                label="Consulta en Lenguaje Natural (Objeto Contractual)",
                placeholder="Ej. Mantenimiento y mejoramiento de vías terciarias en zona rural...",
                value="Mantenimiento y mejoramiento de vías terciarias en zona rural",
                scale=4,
            )
            btn_buscar = gr.Button("Buscar Semánticamente", variant="primary", scale=1)

        with gr.Row():
            depto_dd = gr.Dropdown(
                choices=motor.departamentos_disponibles, value="TODOS", label="Departamento"
            )
            anio_dd = gr.Dropdown(
                choices=motor.anios_disponibles, value="TODOS", label="Año de Firma"
            )
            val_min_num = gr.Number(value=0, label="Valor Mínimo (Millones COP)")
            val_max_num = gr.Number(value=0, label="Valor Máximo (0 = Sin límite)")

        with gr.Row():
            excluir_ps_chk = gr.Checkbox(
                value=True,
                label="Excluir Prestación de Servicios (Saneamiento Modalidad + Segmento UNSPSC 80)",
            )
            top_k_slider = gr.Slider(minimum=5, maximum=25, value=10, step=1, label="Top-K Contratos")
            usar_llm_chk = gr.Checkbox(
                value=False,
                label="Activar Síntesis Generativa Local (Qwen2.5-3B-4bit en Metal GPU)",
            )

        telemetria_out = gr.Markdown()
        resumen_rag_out = gr.Markdown()

        tabla_resultados = gr.Dataframe(
            headers=[
                "#", "Similitud", "Proceso SECOP II", "Depto (Año)",
                "Entidad Contratante", "Modalidad", "Clase UNSPSC", "Valor (COP)", "Objeto Contractual"
            ],
            datatype=["number", "str", "str", "str", "str", "str", "str", "str", "str"],
            wrap=True,
            label="Contratos Recuperados por Proximidad Semántica",
        )

        gr.Examples(
            examples=[
                ["Mantenimiento y mejoramiento de vías terciarias en zona rural", "CUNDINAMARCA", "TODOS", 0, 0, True, 10, False],
                ["Suministro de elementos de papelería, útiles escolares y material didáctico", "TODOS", "2022", 0, 0, True, 10, False],
                ["Adquisición y mantenimiento de equipos biomédicos y ambulancias para hospital", "ANTIOQUIA", "TODOS", 0, 0, True, 10, False],
                ["Desarrollo de software, analítica de datos e infraestructura tecnológica en la nube", "TODOS", "2024", 50, 0, True, 10, False],
            ],
            inputs=[
                consulta_input, depto_dd, anio_dd, val_min_num, val_max_num,
                excluir_ps_chk, top_k_slider, usar_llm_chk
            ],
        )

        btn_buscar.click(
            fn=motor.buscar_y_sintetizar,
            inputs=[
                consulta_input, depto_dd, anio_dd, val_min_num, val_max_num,
                excluir_ps_chk, top_k_slider, usar_llm_chk
            ],
            outputs=[resumen_rag_out, telemetria_out, tabla_resultados],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor Web RAG SECOP-Vec (8 GB RAM)")
    parser.add_argument(
        "--sin-pre-descarga-llm",
        action="store_true",
        help="Omitir la descarga inicial de Qwen2.5-3B-4bit y descargarlo solo si se activa en la UI.",
    )
    args = parser.parse_args()

    verificar_y_descargar_componentes(incluir_llm_mlx=not args.sin_pre_descarga_llm)
    motor_secop = MotorWebSECOPVecRAG()
    interfaz = construir_interfaz_gradio(motor_secop)
    interfaz.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
