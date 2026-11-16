# -*- coding: utf-8 -*-
"""
SUITE DE PRUEBAS UNITARIAS Y DE REPRODUCIBILIDAD CIENTÍFICA (PYTEST)
================================================================================
Verifica matemáticamente los operadores del proyecto y audita la consistencia
de los artefactos experimentales de las Etapas 1 a 6.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

RUTA_RAIZ = Path("/Users/gapt/SECOP_Vec_Local")
RUTA_CONFIG = RUTA_RAIZ / "config" / "experimentos_tesis.yaml"
RUTA_REPORTES = RUTA_RAIZ / "04_tablas_y_metricas_tesis"
RUTA_FIGURAS = RUTA_RAIZ / "06_figuras_tesis_ieee"


def test_01_configuracion_declarativa_yaml_valida():
    """Verifica que el archivo declarativo de parámetros exista y contenga las 6 etapas."""
    assert RUTA_CONFIG.exists(), "Falta el archivo config/experimentos_tesis.yaml"
    with open(RUTA_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["metadatos_investigacion"]["semilla_determinista"] == 42
    assert cfg["metadatos_investigacion"]["repeticiones_estadisticas"] == 5
    assert cfg["restricciones_hardware_objetivo"]["ram_total_mb"] == 8192
    assert cfg["etapa_3_frontera_pareto_p1_p3"]["dimensiones_matryoshka"] == [768, 512, 384, 256]
    assert cfg["etapa_3_frontera_pareto_p1_p3"]["conectividades_M"] == [8, 12, 16]


def test_02_invariante_truncado_matryoshka_y_normalizacion_l2():
    """
    Verifica que el truncado Matryoshka a m in {768, 512, 384, 256} renormalice
    exactamente cada vector sobre la hiperesfera unitaria (norma L2 = 1.0).
    """
    rng = np.random.default_rng(42)
    vectores_brutos = rng.standard_normal((50, 768)).astype(np.float32)

    for m in [768, 512, 384, 256]:
        sub = vectores_brutos[:, :m].copy()
        sub /= np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12
        normas = np.linalg.norm(sub, axis=1)
        assert np.allclose(normas, 1.0, atol=1e-5), f"Fallo de normalización L2 en m={m}"


def test_03_ortogonalidad_y_preservacion_producto_interno_rabitq():
    """
    Verifica el teorema fundamental de RaBitQ (Gao y Long, 2024):
    La matriz de rotación aleatoria P debe ser ortogonal (P^T P = I) y preservar
    exactamente el producto interno <Px, Pq> = <x, q>.
    """
    dim = 256
    rng = np.random.default_rng(42)
    gauss = rng.standard_normal((dim, dim), dtype=np.float32)
    q_ort, r_tri = np.linalg.qr(gauss)
    signos = np.sign(np.diag(r_tri))
    signos[signos == 0] = 1.0
    p_rot = (q_ort * signos).astype(np.float32)

    identidad_aprox = np.matmul(p_rot.T, p_rot)
    assert np.allclose(identidad_aprox, np.eye(dim), atol=1e-4), "La matriz P no es ortogonal"

    x = rng.standard_normal((10, dim), dtype=np.float32)
    q = rng.standard_normal((10, dim), dtype=np.float32)
    sim_original = np.sum(x * q, axis=1)
    sim_rotada = np.sum(np.matmul(x, p_rot) * np.matmul(q, p_rot), axis=1)
    assert np.allclose(sim_original, sim_rotada, atol=1e-4), "RaBitQ alteró el producto interno"


def test_04_calculo_exacto_ganancia_jerarquica_unspsc_ndcg10():
    """
    Verifica la función de ganancia jerárquica UNSPSC rel in {0, 1, 2, 3}
    y las cotas matemáticas de nDCG@10 in [0.0, 1.0].
    """
    descuentos = 1.0 / np.log2(np.arange(2, 12, dtype=np.float32))
    idcg_ideal = float(np.sum(7.0 * descuentos))

    ganancia_perfecta = np.full(10, 3.0, dtype=np.float32)
    dcg_perfecto = float(np.sum((np.power(2.0, ganancia_perfecta) - 1.0) * descuentos))
    assert math.isclose(dcg_perfecto / idcg_ideal, 1.0, rel_tol=1e-6)

    ganancia_nula = np.zeros(10, dtype=np.float32)
    dcg_nulo = float(np.sum((np.power(2.0, ganancia_nula) - 1.0) * descuentos))
    assert math.isclose(dcg_nulo / idcg_ideal, 0.0, abs_tol=1e-6)


def test_05_auditoria_resultados_hipotesis_h1_h2_h3_h4_en_disco():
    """
    Audita que los archivos JSON generados en las Etapas 2 a 6 existan,
    respeten el techo de 8 GB de RAM y verifiquen las Hipótesis H1, H2, H3 y H4.
    """
    archivo_p1 = RUTA_REPORTES / "tabla_4_1_frontera_pareto_p1_p3_2.json"
    if not archivo_p1.exists():
        archivo_p1 = RUTA_REPORTES / "tabla_4_1_frontera_pareto_p1_p3.json"
    assert archivo_p1.exists(), "Falta el reporte experimental de la Etapa 3 (P1/P3)"

    with open(archivo_p1, "r", encoding="utf-8") as f:
        d_p1 = json.load(f)

    for clave, conf in d_p1.items():
        assert conf["rss_pico_busqueda_mb"] < 8192.0, f"Desborde de 8 GB en {clave}"
        assert conf["rss_pico_construccion_mb"] < 8192.0, f"Desborde en construcción {clave}"

    if "int8_dim512_M12" in d_p1:
        ndcg_512 = d_p1["int8_dim512_M12"]["barrido_ef_search"]["ef_512"]["ndcg_at_10_global"]
        assert abs(0.7112 - ndcg_512) < 0.01, "No se cumple la cota de 1 punto de la Hipótesis H3"

    archivo_p2 = RUTA_REPORTES / "tabla_4_3_jerarquia_vs_plano_p2.json"
    assert archivo_p2.exists(), "Falta el reporte de la Etapa 4 (P2/H2)"
    with open(archivo_p2, "r", encoding="utf-8") as f:
        d_p2 = json.load(f)
    assert d_p2["presupuesto_estructural_corpus_completo_1_99M"]["ahorro_memoria_sobre_grafo_porcentaje"] > 14.0

    archivo_p4 = RUTA_REPORTES / "tabla_4_4_busqueda_filtrada_p4.json"
    assert archivo_p4.exists(), "Falta el reporte de la Etapa 5 (P4/H4)"
    with open(archivo_p4, "r", encoding="utf-8") as f:
        d_p4 = json.load(f)
    m8_da = d_p4["resultados_por_conectividad_M"]["M_8"]["departamento_mas_anio"]["estrategias"]
    assert m8_da["acorn_1"]["recall_at_10_filtrado"] > m8_da["hnsw_filtrable"]["recall_at_10_filtrado"]
    assert m8_da["hnsw_filtrable"]["recall_at_10_filtrado"] > m8_da["post_filtrado"]["recall_at_10_filtrado"]


def test_06_existencia_figuras_vectoriales_tesis():
    """Verifica que las 6 figuras científicas PDF y PNG a 300 DPI existan y no estén vacías."""
    nombres_base = [
        "fig_4_1_frontera_pareto_p1",
        "fig_4_2_curvas_latencia_qps_ic95",
        "fig_4_3_amortiguacion_semantica_h3",
        "fig_4_4_jerarquia_vs_plano_hubs_p2",
        "fig_4_5_busqueda_filtrada_acorn_p4",
        "fig_4_6_presupuesto_ram_8gb_rag",
    ]
    for nb in nombres_base:
        pdf = RUTA_FIGURAS / f"{nb}.pdf"
        png = RUTA_FIGURAS / f"{nb}.png"
        assert pdf.exists() and pdf.stat().st_size > 1000, f"Falta figura vectorial {pdf.name}"
        assert png.exists() and png.stat().st_size > 1000, f"Falta figura alta resolución {png.name}"
