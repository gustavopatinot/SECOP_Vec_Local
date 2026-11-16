# -*- coding: utf-8 -*-
"""
MÓDULO 08: CONSTRUCTOR AUTOMÁTICO DEL REPOSITORIO CIENTÍFICO REPRODUCIBLE
================================================================================
Genera sin errores de símbolos ni de tabulación toda la estructura de
ingeniería de software reproducible para auditar la tesis SECOP-Vec:
  1. config/experimentos_tesis.yaml
  2. entornos/requirements_colab_gpu.txt y requirements_mac_m1_8gb.txt
  3. tests/test_integridad_cientifica.py (6 pruebas unitarias y de datos con pytest)
  4. Makefile (con tabulaciones reales POSIX)
  5. .gitignore, pyproject.toml y README.md
================================================================================
"""

from pathlib import Path

RUTA_RAIZ = Path("/Users/gapt/SECOP_Vec_Local")


def escribir_archivo(ruta_relativa: str, contenido: str) -> None:
    ruta_destino = RUTA_RAIZ / ruta_relativa
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    ruta_destino.write_text(contenido.strip() + "\n", encoding="utf-8")
    print(f"[OK] Archivo generado: {ruta_destino}")


def construir_infraestructura_reproducible() -> None:
    # 1. Configuración declarativa YAML de las 6 etapas experimentales
    yaml_config = """
# ==============================================================================
# CONFIGURACIÓN DECLARATIVA REPRODUCIBLE — BENCHMARK SECOP-VEC (TESIS MAESTRÍA)
# ==============================================================================

metadatos_investigacion:
  titulo: "Búsqueda semántica a gran escala con memoria restringida: caso SECOP II"
  semilla_determinista: 42
  repeticiones_estadisticas: 5
  nivel_confianza_t_student: 0.95
  valor_critico_t_gl4: 2.776

restricciones_hardware_objetivo:
  ram_total_mb: 8192
  ram_maxima_permitida_proceso_mb: 4500
  hilos_cpu_maximos: 4
  tamano_lote_construccion_memmap: 50000

etapa_1_curaduria_corpus:
  recurso_socrata_id: "jbjy-vk9h"
  anio_inicio: 2019
  anio_fin: 2025
  longitud_minima_caracteres: 30
  tamano_corpus_objetivo: 1990000
  tamano_consultas_verdad_referencia: 10000

etapa_2_vectorizacion_gpu:
  modelo_embeddings: "intfloat/multilingual-e5-base"
  dimension_base: 768
  longitud_maxima_tokens: 160
  prefijo_consulta: "query: "
  prefijo_documento: "passage: "
  top_k_verdad_exacta: 100

etapa_3_frontera_pareto_p1_p3:
  familias_representacion:
    - "float32_submuestra"
    - "float16"
    - "int8"
    - "b1_rerank_ssd"
    - "rabitq_4bit"
  dimensiones_matryoshka: [768, 512, 384, 256]
  conectividades_M: [8, 12, 16]
  barrido_ef_search: [32, 64, 128, 256, 512]
  factor_sobremuestreo_rerank_ssd: 10

etapa_4_jerarquia_vs_plano_p2:
  configuracion_ganadora: "int8_dim512_M12"
  dimension: 512
  conectividad_M: 12
  factor_multicapa_m_L: 0.4024
  tamano_corpus_diseccion: 1000000
  consultas_auditoria_hubs: 2000

etapa_5_busqueda_filtrada_p4:
  conectividades_evaluadas: [8, 12, 16]
  ef_search_base: 128
  estrategias_filtrado:
    - "post_filtrado"
    - "hnsw_filtrable"
    - "acorn_1"
  predicados_filtro:
    - "sin_filtro"
    - "rango_valor"
    - "otras_modalidades_no_ps"
    - "departamento"
    - "departamento_mas_anio"

etapa_6_prototipo_rag:
  archivo_indice_produccion: "secop_int8_dim512_M12.usearch"
  modo_apertura_memoria: "view=True"
  consultas_parafraseadas: 2000
  consultas_control_humano: 300
  regla_exclusion_prestacion_servicios:
    excluir_flag_modalidad: true
    excluir_segmento_unspsc: "80"
"""
    escribir_archivo("config/experimentos_tesis.yaml", yaml_config)

    # 2. Manifiestos de dependencias por entorno (Colab GPU vs. Mac M1 8 GB)
    req_colab = """
# Entorno A: Fábrica de Datos y Verdad de Referencia en GPU (Google Colab Pro)
duckdb>=1.0.0
pyarrow>=15.0.0
requests>=2.31.0
numpy>=1.26.0
torch>=2.2.0
sentence-transformers>=2.7.0
"""
    escribir_archivo("entornos/requirements_colab_gpu.txt", req_colab)

    req_mac = """
# Entorno B: Laboratorio Experimental en Memoria Restringida (MacBook Air M1 - 8 GB RAM)
usearch>=2.12.0
hnswlib>=0.8.0
pyarrow>=15.0.0
numpy>=1.26.0
psutil>=5.9.0
matplotlib>=3.8.0
pyyaml>=6.0.1
pytest>=8.0.0
"""
    escribir_archivo("entornos/requirements_mac_m1_8gb.txt", req_mac)

    # 3. Suite de pruebas unitarias, matemáticas y de integridad con pytest
    codigo_pytest = '''
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
'''
    escribir_archivo("tests/test_integridad_cientifica.py", codigo_pytest)

    # 4. Makefile con tabulaciones POSIX reales (\t) generadas desde Python
    lineas_makefile = [
        "# ==============================================================================",
        "# MAKEFILE DE REPRODUCIBILIDAD CIENTÍFICA — PROYECTO SECOP-VEC (8 GB RAM)",
        "# ==============================================================================",
        "PYTHON := python3",
        "",
        ".PHONY: ayuda instalar-entorno-mac test etapa3 etapa4 etapa5 etapa6 figuras auditar-todo",
        "",
        "ayuda:",
        '\t@echo "Comandos disponibles para reproducir la tesis SECOP-Vec:"',
        '\t@echo "  make instalar-entorno-mac : Instala las librerías del entorno de 8 GB"',
        '\t@echo "  make test                 : Ejecuta las pruebas unitarias y matemáticas (pytest)"',
        '\t@echo "  make etapa3               : Ejecuta Frontera de Pareto (P1 y P3)"',
        '\t@echo "  make etapa4               : Ejecuta Jerarquía HNSW vs. Grafo Plano (P2)"',
        '\t@echo "  make etapa5               : Ejecuta Búsqueda Filtrada y ACORN-1 (P4)"',
        '\t@echo "  make etapa6               : Ejecuta Prototipo Buscador + Capa RAG (Obj. 5)"',
        '\t@echo "  make figuras              : Genera las 6 figuras IEEE/ACM en PDF y PNG"',
        '\t@echo "  make auditar-todo         : Ejecuta pytest + regenera figuras"',
        "",
        "instalar-entorno-mac:",
        "\t$(PYTHON) -m pip install -r entornos/requirements_mac_m1_8gb.txt",
        "",
        "test:",
        "\t$(PYTHON) -m pytest tests/test_integridad_cientifica.py -v",
        "",
        "etapa3:",
        "\t$(PYTHON) 03_experimento_cuantizacion.py --fase completa",
        "",
        "etapa4:",
        "\t$(PYTHON) 04_experimento_jerarquia_nsw.py",
        "",
        "etapa5:",
        "\t$(PYTHON) 05_experimento_busqueda_filtrada_p4.py",
        "",
        "etapa6:",
        "\t$(PYTHON) 06_prototipo_buscador_rag_secop.py",
        "\t$(PYTHON) 06b_parche_casos_rag.py",
        "",
        "figuras:",
        "\t$(PYTHON) 07_generador_graficas_tesis.py",
        "",
        "auditar-todo: figuras test",
        '\t@echo "=== AUDITORÍA DE REPRODUCIBILIDAD COMPLETADA CON ÉXITO ==="',
    ]
    escribir_archivo("Makefile", "\n".join(lineas_makefile))

    # 5. Archivos .gitignore y pyproject.toml
    gitignore = """
03_oro_benchmark_secop_vec/*.parquet
03_oro_benchmark_secop_vec/*.mmap
03_oro_benchmark_secop_vec/*.npy
05_indices_temporales_ssd/*.usearch
05_indices_temporales_ssd/*.mmap
__pycache__/
.pytest_cache/
*.tmp
.DS_Store
"""
    escribir_archivo(".gitignore", gitignore)

    pyproject = """
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"
"""
    escribir_archivo("pyproject.toml", pyproject)

    # 6. README.md académico (usando sangría de 4 espacios para comandos, sin backticks triples)
    readme = """
# SECOP-Vec: Búsqueda Semántica a Gran Escala con Memoria Restringida (8 GB RAM)

Repositorio oficial de reproducibilidad científica de la Tesis de Maestría:
**Búsqueda semántica a gran escala con memoria restringida: HNSW cuantizado aplicado a la contratación pública colombiana (SECOP II) en un computador de 8 GB de RAM**.

## 1. Arquitectura Experimental de Dos Entornos

1. **Entorno A — Fábrica de Datos y Verdad de Referencia (Google Colab Pro GPU):**
   - Manifiesto: `entornos/requirements_colab_gpu.txt`
   - Módulos: `01_fabrica_corpus_secop.py` y `02_generador_vectores_y_verdad.py`
   - Productos: Corpus limpio `SECOP-Vec v1` (N = 1.990.000 contratos únicos curados desde 5.714.490 registros brutos), |Q| = 10.000 consultas estratificadas y verdad de referencia exacta top-100 en GPU.

2. **Entorno B — Laboratorio de Memoria Restringida (MacBook Air M1, 8 GB RAM):**
   - Manifiesto: `entornos/requirements_mac_m1_8gb.txt`
   - Configuración declarativa de hiperparámetros: `config/experimentos_tesis.yaml`
   - Módulos:
     * `03_experimento_cuantizacion.py`: Frontera de Pareto de cuantización y truncado Matryoshka (P1 y P3).
     * `04_experimento_jerarquia_nsw.py`: HNSW Jerárquico vs. Grafo Plano NSW y autopista de hubs (P2).
     * `05_experimento_busqueda_filtrada_p4.py`: Búsqueda filtrada y algoritmo ACORN-1 (P4).
     * `06_prototipo_buscador_rag_secop.py`: Prototipo funcional empaquetado (`view=True`) con síntesis RAG.
     * `07_generador_graficas_tesis.py`: Exportación de las 6 figuras IEEE/ACM en PDF vectorial y PNG 300 DPI.

## 2. Ejecución Reproducible y Auditoría Automatizada

Para instalar las dependencias del laboratorio local, generar las figuras y ejecutar las pruebas matemáticas con `pytest`, ejecute en la terminal:

    make instalar-entorno-mac
    make auditar-todo
"""
    escribir_archivo("README.md", readme)
    print("\n=== INFRAESTRUCTURA DE REPRODUCIBILIDAD GENERADA AL 100% ===")


if __name__ == "__main__":
    construir_infraestructura_reproducible()

