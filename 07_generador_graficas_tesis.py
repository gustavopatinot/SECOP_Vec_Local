# -*- coding: utf-8 -*-
"""
MÓDULO 07: GENERADOR AUTOMÁTICO DE FIGURAS CIENTÍFICAS (ESTILO IEEE/ACM EN ESPAÑOL)
================================================================================
Exporta las 6 figuras maestras del Capítulo 4 y Capítulo 5 en formatos PDF
vectorial y PNG de alta resolución (300 DPI) a partir de los JSON experimentales.
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
registro = logging.getLogger("GeneradorGraficasTesis")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

RUTA_BASE = Path("/Users/gapt/SECOP_Vec_Local")
RUTA_REPORTES = RUTA_BASE / "04_tablas_y_metricas_tesis"
RUTA_FIGURAS = RUTA_BASE / "06_figuras_tesis_ieee"


def cargar_json_mas_completo(nombres_posibles: List[str], respaldo: Dict) -> Dict:
    """Carga el archivo JSON que contenga mayor cantidad de registros entre los candidatos."""
    mejor_diccionario = None
    mejor_nombre = ""
    for nombre in nombres_posibles:
        ruta = RUTA_REPORTES / nombre
        if ruta.exists():
            with open(ruta, "r", encoding="utf-8") as f:
                actual = json.load(f)
            if mejor_diccionario is None or len(actual) > len(mejor_diccionario):
                mejor_diccionario = actual
                mejor_nombre = ruta.name
    if mejor_diccionario is not None:
        registro.info(f"Archivo experimental cargado: {mejor_nombre} ({len(mejor_diccionario)} entradas)")
        return mejor_diccionario
    registro.info(f"Usando respaldo en memoria para: {nombres_posibles[0]}")
    return respaldo


def guardar_figura_doble_formato(fig: plt.Figure, nombre_base: str) -> None:
    """Exporta cada figura en PDF vectorial y PNG a 300 DPI."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta_pdf = RUTA_FIGURAS / f"{nombre_base}.pdf"
    ruta_png = RUTA_FIGURAS / f"{nombre_base}.png"
    fig.savefig(ruta_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(ruta_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    registro.info(f"Figura exportada: {ruta_pdf.name} y {ruta_png.name}")


def generar_fig_4_1_frontera_pareto(datos_p1: Dict, techo_ndcg_gpu: float) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    colores = {
        "float16": "#1f77b4",
        "int8": "#2ca02c",
        "rabitq_4bit": "#9467bd",
        "b1_rerank_ssd": "#d62728",
        "float32_submuestra": "#7f7f7f",
    }
    etiquetas_fam = {
        "float16": "float16 (D=768, M=12)",
        "int8": "int8 Matryoshka (D=768..256, M=12)",
        "rabitq_4bit": "RaBitQ 4-bit + Rotación (D=768, M=12)",
        "b1_rerank_ssd": "Binaria 1-bit + Rerank SSD (D=768..256, M=12)",
        "float32_submuestra": "float32 Submuestra 500k (D=768, M=12)",
    }
    marcadores = {
        "float16": "D",
        "int8": "o",
        "rabitq_4bit": "P",
        "b1_rerank_ssd": "s",
        "float32_submuestra": "^",
    }

    familias_orden = ["float16", "int8", "rabitq_4bit", "b1_rerank_ssd", "float32_submuestra"]
    for fam in familias_orden:
        puntos = []
        for _, conf in datos_p1.items():
            if conf["familia_representacion"] == fam and conf["conectividad_M"] == 12:
                ef512 = conf["barrido_ef_search"]["ef_512"]
                puntos.append((
                    conf["tamano_indice_disco_mb"],
                    ef512["recall_at_10"],
                    ef512["ndcg_at_10_global"],
                    conf["dimension"],
                ))
        if not puntos:
            continue
        puntos.sort(key=lambda item: item[0])
        xs = [p[0] for p in puntos]
        ys_rec = [p[1] for p in puntos]
        ys_ndcg = [p[2] for p in puntos]

        estilo_linea = "-" if len(puntos) > 1 else "None"
        ax1.plot(
            xs, ys_rec, marker=marcadores[fam], linestyle=estilo_linea,
            color=colores[fam], linewidth=2.0, markersize=7, label=etiquetas_fam[fam]
        )
        ax2.plot(
            xs, ys_ndcg, marker=marcadores[fam], linestyle=estilo_linea,
            color=colores[fam], linewidth=2.0, markersize=7, label=etiquetas_fam[fam]
        )

        for x_val, y_r, y_n, dim_val in puntos:
            if fam in ("int8", "b1_rerank_ssd"):
                ax1.annotate(
                    f"{dim_val}D", (x_val, y_r), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, fontweight="bold", color=colores[fam]
                )
                ax2.annotate(
                    f"{dim_val}D", (x_val, y_n), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, fontweight="bold", color=colores[fam]
                )

    ax1.set_title("(a) Fidelidad Geométrica: Memoria vs. Recall@10 (ef=512)")
    ax1.set_xlabel("Tamaño del Índice en Disco / Mapeo Memmap (MB)")
    ax1.set_ylabel("Recall@10 contra Búsqueda Exacta GPU")
    ax1.set_ylim(0.15, 1.05)
    ax1.legend(loc="lower right", frameon=True)

    ax2.axhline(
        techo_ndcg_gpu, color="#000000", linestyle="-.", linewidth=1.5,
        label=f"Techo Exacto GPU (nDCG@10 = {techo_ndcg_gpu:.4f})"
    )
    ax2.axhspan(
        techo_ndcg_gpu - 0.0112, techo_ndcg_gpu, color="#2ca02c", alpha=0.12,
        label="Zona de Pérdida Imperceptible (< 1,1 pp nDCG@10 - H3)"
    )
    ax2.set_title("(b) Relevancia de Usuario: Memoria vs. nDCG@10 UNSPSC (ef=512)")
    ax2.set_xlabel("Tamaño del Índice en Disco / Mapeo Memmap (MB)")
    ax2.set_ylabel("nDCG@10 Jerárquico por Código UNSPSC")
    ax2.set_ylim(0.53, 0.73)
    ax2.legend(loc="lower right", frameon=True)

    fig.suptitle(
        "Figura 4.1. Frontera de Pareto Calidad × Memoria sobre N = 1.990.000 Contratos del SECOP II (Hipótesis H1 y H3)",
        y=1.02, fontweight="bold"
    )
    guardar_figura_doble_formato(fig, "fig_4_1_frontera_pareto_p1")


def generar_fig_4_2_curvas_velocidad_relevancia(datos_p1: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    configs_clave = [
        ("float16_dim768_M12", "float16 (768D, M=12)", "#1f77b4", "D"),
        ("int8_dim768_M12", "int8 (768D, M=12)", "#17becf", "v"),
        ("int8_dim512_M12", "int8 Óptimo Pareto (512D, M=12)", "#2ca02c", "o"),
        ("int8_dim384_M12", "int8 Sub-GB (384D, M=12)", "#ff7f0e", "^"),
        ("rabitq_4bit_dim768_M12", "RaBitQ 4-bit (768D, M=12)", "#9467bd", "P"),
        ("b1_rerank_ssd_dim768_M12", "Binaria 1-bit + SSD (768D, M=12)", "#d62728", "s"),
    ]
    efs = ["ef_64", "ef_128", "ef_256", "ef_512"]

    for clave, etiqueta, color, marcador in configs_clave:
        if clave not in datos_p1:
            continue
        barrido = datos_p1[clave]["barrido_ef_search"]
        ndcgs = [barrido[e]["ndcg_at_10_global"] for e in efs]
        qps_m = [barrido[e]["qps_media"] for e in efs]
        qps_ic = [barrido[e]["qps_ic95"] for e in efs]
        lat_m = [barrido[e]["latencia_p50_ms_media"] for e in efs]
        lat_ic = [barrido[e]["latencia_p50_ms_ic95"] for e in efs]

        ax1.errorbar(
            ndcgs, qps_m, yerr=qps_ic, label=etiqueta, color=color,
            marker=marcador, linewidth=2.0, markersize=6, capsize=3
        )
        ax2.errorbar(
            lat_m, ndcgs, xerr=lat_ic, label=etiqueta, color=color,
            marker=marcador, linewidth=2.0, markersize=6, capsize=3
        )

    ax1.set_yscale("log")
    ax1.set_title("(a) Caudal de Consultas (QPS, escala log) vs. Relevancia nDCG@10")
    ax1.set_xlabel("nDCG@10 Jerárquico (UNSPSC)")
    ax1.set_ylabel("Consultas por Segundo (QPS ± IC 95%)")
    ax1.legend(loc="lower left", frameon=True)

    ax2.set_title("(b) Latencia Mediana p50 (ms ± IC 95%) vs. Relevancia nDCG@10")
    ax2.set_xlabel("Latencia p50 por Consulta en MacBook Air M1 (ms)")
    ax2.set_ylabel("nDCG@10 Jerárquico (UNSPSC)")
    ax2.legend(loc="lower right", frameon=True)

    fig.suptitle(
        "Figura 4.2. Frontera Velocidad × Relevancia con Intervalos de Confianza al 95% (ef_search de 64 a 512)",
        y=1.02, fontweight="bold"
    )
    guardar_figura_doble_formato(fig, "fig_4_2_curvas_latencia_qps_ic95")


def generar_fig_4_3_amortiguacion_semantica_h3(datos_p1: Dict, techo_ndcg_gpu: float) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    orden_configs = [
        ("float16_dim768_M12", "float16\n768D"),
        ("int8_dim768_M16", "int8\n768D (M16)"),
        ("int8_dim768_M12", "int8\n768D (M12)"),
        ("int8_dim512_M12", "int8\n512D (M12)"),
        ("int8_dim384_M12", "int8\n384D (M12)"),
        ("int8_dim256_M12", "int8\n256D (M12)"),
    ]

    nombres = []
    perdida_recall_pp = []
    perdida_ndcg_pp = []
    ndcg_ps = []
    ndcg_otras = []
    ndcg_glob = []

    for clave, etiq in orden_configs:
        if clave not in datos_p1:
            continue
        ef512 = datos_p1[clave]["barrido_ef_search"]["ef_512"]
        nombres.append(etiq)
        perdida_recall_pp.append((1.0 - ef512["recall_at_10"]) * 100.0)
        perdida_ndcg_pp.append((techo_ndcg_gpu - ef512["ndcg_at_10_global"]) * 100.0)
        ndcg_ps.append(ef512["ndcg_at_10_prestacion_servicios"])
        ndcg_otras.append(ef512["ndcg_at_10_otras_modalidades"])
        ndcg_glob.append(ef512["ndcg_at_10_global"])

    x = np.arange(len(nombres))
    ancho = 0.36

    b1 = ax1.bar(
        x - ancho / 2, perdida_recall_pp, ancho,
        label="Pérdida Técnica de Recall@10 (pp)", color="#e6550d", alpha=0.88
    )
    b2 = ax1.bar(
        x + ancho / 2, perdida_ndcg_pp, ancho,
        label="Pérdida Real de Relevancia nDCG@10 (pp)", color="#2ca02c", alpha=0.92
    )

    for barra in b1:
        h = barra.get_height()
        ax1.annotate(
            f"-{h:.1f} pp", (barra.get_x() + barra.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8
        )
    for barra in b2:
        h = barra.get_height()
        ax1.annotate(
            f"-{h:.2f} pp", (barra.get_x() + barra.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold"
        )

    ax1.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.3, label="Umbral Hipótesis H3 (1,0 pp)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres)
    ax1.set_ylim(0, 36)
    ax1.set_ylabel("Pérdida respecto a Fuerza Bruta Exacta (Puntos Porcentuales)")
    ax1.set_title("(a) Amortiguación Semántica: Caída de Recall@10 vs. Caída de nDCG@10")
    ax1.legend(loc="upper left", frameon=True)

    ax2.plot(x, ndcg_ps, marker="o", linewidth=2.2, color="#1f77b4", label="Prestación de Servicios (81,51% del corpus)")
    ax2.plot(x, ndcg_glob, marker="s", linewidth=2.2, color="#2ca02c", label="Global SECOP-Vec (1.990.000 contratos)")
    ax2.plot(x, ndcg_otras, marker="^", linewidth=2.2, color="#d62728", label="Otras Modalidades sin Filtro (18,49% del corpus)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres)
    ax2.set_ylim(0.30, 0.85)
    ax2.set_ylabel("nDCG@10 Jerárquico por Código UNSPSC")
    ax2.set_title("(b) Desempeño Desglosado por Modalidad Contractual (Justificación de P4)")
    ax2.legend(loc="center right", frameon=True)

    fig.suptitle(
        "Figura 4.3. Contraste entre Pérdida Geométrica (Recall@10) y Utilidad Percibida (nDCG@10) — Validación de H3",
        y=1.02, fontweight="bold"
    )
    guardar_figura_doble_formato(fig, "fig_4_3_amortiguacion_semantica_h3")


def generar_fig_4_4_jerarquia_vs_plano_p2(datos_p2: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    dist_capas = datos_p2["presupuesto_estructural_corpus_completo_1_99M"]["distribucion_nodos_por_capa"]
    capas = list(dist_capas.keys())
    conteos = [dist_capas[c] for c in capas]

    barras = ax1.bar(capas, conteos, color="#3182bd", edgecolor="black", alpha=0.88)
    ax1.set_yscale("log")
    for b, c in zip(barras, conteos):
        ax1.annotate(
            f"{c:,}", (b.get_x() + b.get_width() / 2, c),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold"
        )
    ax1.set_ylim(0.5, 8_000_000)
    ax1.set_xlabel("Nivel de la Jerarquía HNSW (M = 12, m_L = 0,4024)")
    ax1.set_ylabel("Número de Nodos Presentes (Escala Logarítmica)")
    ax1.set_title("(a) Pirámide Multicapa HNSW (Ahorro Grafo Plano = 14,85% / 33,09 MB)")

    hubs = datos_p2["analisis_autopista_hubs_L0"]
    porcentajes_nodos = ["Top 1% Hubs", "Top 5% Hubs", "Top 10% Hubs"]
    trafico_real = [
        hubs["trafico_concentrado_en_top_1_pct_hubs"],
        hubs["trafico_concentrado_en_top_5_pct_hubs"],
        hubs["trafico_concentrado_en_top_10_pct_hubs"],
    ]
    trafico_uniforme = [1.0, 5.0, 10.0]

    x = np.arange(len(porcentajes_nodos))
    ancho = 0.35
    ax2.bar(x - ancho / 2, trafico_uniforme, ancho, label="Grafo sin Hubs (Tráfico Uniforme)", color="#bdbdbd")
    b_real = ax2.bar(x + ancho / 2, trafico_real, ancho, label="Capa Base L0 SECOP-Vec (Autopista de Hubs)", color="#2ca02c")

    for b in b_real:
        h = b.get_height()
        ax2.annotate(
            f"{h:.2f}%", (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold"
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(porcentajes_nodos)
    ax2.set_ylim(0, 26)
    ax2.set_ylabel("Porcentaje de Accesos del Haz de Búsqueda en L0 (%)")
    ax2.set_title("(b) Formación de la «Autopista de Hubs» en Español Administrativo (H2)")
    ax2.legend(loc="upper left", frameon=True)

    fig.suptitle(
        "Figura 4.4. Disección de la Jerarquía HNSW frente al Grafo Plano NSW y Concentración de Nodos Hub (Pregunta P2)",
        y=1.02, fontweight="bold"
    )
    guardar_figura_doble_formato(fig, "fig_4_4_jerarquia_vs_plano_hubs_p2")


def generar_fig_4_5_busqueda_filtrada_p4(datos_p4: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    conectividades = ["M_8", "M_12", "M_16"]
    etiq_m = ["M = 8\n(Grafo Compacto)", "M = 12\n(Equilibrio Pareto)", "M = 16\n(Grafo Denso)"]

    rec_post = [datos_p4["resultados_por_conectividad_M"][m]["departamento_mas_anio"]["estrategias"]["post_filtrado"]["recall_at_10_filtrado"] for m in conectividades]
    rec_filt = [datos_p4["resultados_por_conectividad_M"][m]["departamento_mas_anio"]["estrategias"]["hnsw_filtrable"]["recall_at_10_filtrado"] for m in conectividades]
    rec_acorn = [datos_p4["resultados_por_conectividad_M"][m]["departamento_mas_anio"]["estrategias"]["acorn_1"]["recall_at_10_filtrado"] for m in conectividades]

    x = np.arange(len(conectividades))
    ancho = 0.25

    ax1.bar(x - ancho, rec_post, ancho, label="Post-filtrado", color="#d62728", alpha=0.85)
    ax1.bar(x, rec_filt, ancho, label="HNSW Filtrable (1 salto)", color="#1f77b4", alpha=0.88)
    ax1.bar(x + ancho, rec_acorn, ancho, label="ACORN-1 (Expansión 2 saltos)", color="#2ca02c", alpha=0.92)

    for i in range(len(x)):
        ax1.annotate(f"{rec_post[i]:.2f}", (x[i] - ancho, rec_post[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax1.annotate(f"{rec_filt[i]:.2f}", (x[i], rec_filt[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax1.annotate(f"{rec_acorn[i]:.2f}", (x[i] + ancho, rec_acorn[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(etiq_m)
    ax1.set_ylim(0.45, 0.96)
    ax1.set_ylabel("Recall@10 Filtrado Exacto")
    ax1.set_title("(a) Filtro Selectivo «Depto + Año» (σ = 2,24%) por Conectividad M")
    ax1.legend(loc="lower right", frameon=True)

    filtros_orden = ["sin_filtro", "rango_valor", "departamento", "departamento_mas_anio", "otras_modalidades_no_ps"]
    etiq_filtros = [
        "Sin Filtro\n(100%)",
        "Rango Valor\n(32,1%)",
        "Departamento\n(13,8%)",
        "Depto + Año\n(2,24%)",
        "No Prest. Serv.\n(18,5% Foso)",
    ]
    m12 = datos_p4["resultados_por_conectividad_M"]["M_12"]
    r_p_12 = [m12[f]["estrategias"]["post_filtrado"]["recall_at_10_filtrado"] for f in filtros_orden]
    r_h_12 = [m12[f]["estrategias"]["hnsw_filtrable"]["recall_at_10_filtrado"] for f in filtros_orden]
    r_a_12 = [m12[f]["estrategias"]["acorn_1"]["recall_at_10_filtrado"] for f in filtros_orden]

    x2 = np.arange(len(filtros_orden))
    ax2.bar(x2 - ancho, r_p_12, ancho, label="Post-filtrado", color="#d62728", alpha=0.85)
    ax2.bar(x2, r_h_12, ancho, label="HNSW Filtrable (1 salto)", color="#1f77b4", alpha=0.88)
    ax2.bar(x2 + ancho, r_a_12, ancho, label="ACORN-1 (2 saltos)", color="#2ca02c", alpha=0.92)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(etiq_filtros)
    ax2.set_ylim(0.15, 0.98)
    ax2.set_ylabel("Recall@10 Filtrado en M = 12")
    ax2.set_title("(b) Desempeño por Tipo de Predicado y Ruptura del «Foso Topológico»")
    ax2.legend(loc="lower left", frameon=True)

    fig.suptitle(
        "Figura 4.5. Evaluación de Estrategias de Búsqueda Filtrada bajo Memoria Restringida (Pregunta P4 / Hipótesis H4)",
        y=1.02, fontweight="bold"
    )
    guardar_figura_doble_formato(fig, "fig_4_5_busqueda_filtrada_acorn_p4")


def generar_fig_4_6_presupuesto_ram_8gb_rag(datos_p6: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    escenarios = [
        "HNSW Estándar\nfloat32 (768D, M16)\n[Sin Comprimir]",
        "Prototipo SECOP-Vec\nint8 (512D, M12)\n[Motor Listo view=True]",
        "Prototipo SECOP-Vec\n+ Síntesis Capa RAG\n[Pico Operativo]",
    ]
    so_base_mb = [2200.0, 2200.0, 2200.0]
    proceso_mb = [
        6370.0,
        datos_p6["presupuesto_memoria_8gb"]["rss_proceso_listo_mb"],
        datos_p6["presupuesto_memoria_8gb"]["rss_pico_con_capa_rag_mb"],
    ]

    x = np.arange(len(escenarios))
    ancho = 0.45
    ax1.bar(x, so_base_mb, ancho, label="macOS + Aplicaciones Base (~2,2 GB)", color="#969696")
    ax1.bar(
        x, proceso_mb, ancho, bottom=so_base_mb,
        label="Motor Vectorial SECOP-Vec + Capa RAG",
        color=["#d62728", "#2ca02c", "#1f77b4"], alpha=0.90
    )

    ax1.axhline(8192.0, color="black", linestyle="--", linewidth=2.0, label="Techo Físico de Hardware (8.192 MB RAM)")
    for i, (s, p) in enumerate(zip(so_base_mb, proceso_mb)):
        total = s + p
        estado = "¡DESBORDE SWAP!" if total > 8192 else f"Total: {total:,.0f} MB\n(Libre: {8192 - total:,.0f} MB)"
        ax1.annotate(
            estado, (x[i], total), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=8.5, fontweight="bold",
            color="#d62728" if total > 8192 else "#000000"
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(escenarios)
    ax1.set_ylim(0, 9800)
    ax1.set_ylabel("Consumo de Memoria RAM Unificada (MB)")
    ax1.set_title("(a) Presupuesto de Memoria Física en Equipo de 8 GB de RAM")
    ax1.legend(loc="upper right", frameon=True)

    val_para = datos_p6["validacion_2000_consultas_parafraseadas"]
    val_300 = datos_p6["validacion_300_consultas_acuerdo_relevancia"]

    metricas_nom = [
        "Precisión@10 Seg.\n(2.000 Parafraseadas)",
        "nDCG@10 Global\n(2.000 Parafraseadas)",
        "Relevancia Exacta\nGPU (300 Control)",
        "Relevancia Compacto\n8 GB (300 Control)",
        "Tasa de Acuerdo\nExacta vs. 8 GB",
    ]
    valores_val = [
        val_para["precision_at_10_segmento_parafraseadas"] * 100.0,
        val_para["ndcg_at_10_parafraseadas"] * 100.0,
        val_300["relevancia_media_busqueda_exacta"] * 100.0,
        val_300["relevancia_media_indice_compacto_8gb"] * 100.0,
        val_300["tasa_acuerdo_con_referencia_pct"],
    ]

    x2 = np.arange(len(metricas_nom))
    colores_b = ["#3182bd", "#3182bd", "#756bb1", "#2ca02c", "#ff7f0e"]
    barras_v = ax2.bar(x2, valores_val, width=0.5, color=colores_b, alpha=0.9)
    for b in barras_v:
        h = b.get_height()
        ax2.annotate(
            f"{h:.2f}%", (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold"
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(metricas_nom, rotation=12)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Porcentaje / Puntaje Normalizado (%)")
    ax2.set_title("(b) Validación en Consultas Parafraseadas y Acuerdo con Referencia")

    fig.suptitle(
        "Figura 4.6. Viabilidad Operativa del Prototipo con Capa RAG en 8 GB y Validación de Relevancia (Objetivos 3 y 5)",
        y=1.02, fontweight="bold"
    )
    guardar_figura_doble_formato(fig, "fig_4_6_presupuesto_ram_8gb_rag")


if __name__ == "__main__":
    datos_etapa2 = cargar_json_mas_completo(
        ["tabla_3_2_verdad_referencia_gpu.json"],
        {"ndcg_at_10_exacto_jerarquico": 0.7112},
    )
    techo_ndcg = float(datos_etapa2.get("ndcg_at_10_exacto_jerarquico", 0.7112))

    datos_etapa3 = cargar_json_mas_completo(
        ["tabla_4_1_frontera_pareto_p1_p3_2.json", "tabla_4_1_frontera_pareto_p1_p3.json"],
        {},
    )
    datos_etapa4 = cargar_json_mas_completo(
        ["tabla_4_3_jerarquia_vs_plano_p2.json"],
        {},
    )
    datos_etapa5 = cargar_json_mas_completo(
        ["tabla_4_4_busqueda_filtrada_p4.json"],
        {},
    )
    datos_etapa6 = cargar_json_mas_completo(
        ["tabla_4_5_validacion_prototipo_rag.json"],
        {},
    )

    generar_fig_4_1_frontera_pareto(datos_etapa3, techo_ndcg)
    generar_fig_4_2_curvas_velocidad_relevancia(datos_etapa3)
    generar_fig_4_3_amortiguacion_semantica_h3(datos_etapa3, techo_ndcg)
    generar_fig_4_4_jerarquia_vs_plano_p2(datos_etapa4)
    generar_fig_4_5_busqueda_filtrada_p4(datos_etapa5)
    generar_fig_4_6_presupuesto_ram_8gb_rag(datos_etapa6)

    print("\n=== LAS 6 FIGURAS CIENTÍFICAS (PDF + PNG 300 DPI) FUERON GENERADAS CON ÉXITO ===")
    print(f"Carpeta de salida: {RUTA_FIGURAS}")
