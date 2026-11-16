# SECOP-Vec: Búsqueda Semántica a Gran Escala con Memoria Restringida (8 GB RAM)

Repositorio oficial de reproducibilidad científica de la Tesis de Maestría:
**«Búsqueda semántica a gran escala con memoria restringida: HNSW cuantizado aplicado a la contratación pública colombiana (SECOP II) en un computador de 8 GB de RAM»**.

---

## 1. Arquitectura Híbrida de Almacenamiento (GitHub + Google Drive)

Debido a las restricciones de tamaño de archivos binarios en GitHub (`>100 MB`), este repositorio versiona el **100 % del código fuente, configuraciones declarativas (`YAML`), suite de pruebas (`pytest`), métricas (`JSON`) y figuras vectoriales**, mientras que los artefactos masivos del benchmark `SECOP-Vec v1` y el índice preconstruido `.usearch` residen de forma persistente en **Google Drive** bajo una arquitectura Medallón:

* **Enlace público al Lago de Datos (Solo Lectura para Evaluadores):** [archivos Google Drive](https://drive.google.com/drive/folders/1ywM6KPOM25WzngQBB7C-OWD5FpUNZbGX?usp=sharing)

| Capa en Google Drive (`Mi unidad/SECOP_Vec_Tesis/`) | Artefactos Almacenados | Volumen | Etapa / Uso Reproducible |
| :--- | :--- | :---: | :--- |
| **`01_bronce_socrata_parquet/`** | `secop_bronce_lote_*.parquet` y `estado_descarga_checkpoint.json` ($5.714.490$ contratos crudos desde `datos.gov.co` recurso `jbjy-vk9h`) |  **1,50 GB** | **Etapa 1** (`01_fabrica_corpus_secop.py` en Colab Pro). |
| **`02_plata_corpus_deduplicado/`** | `secop_plata_deduplicado.parquet` ($2.000.000$ de objetos contractuales únicos deduplicados por hash MD5 y clasificados en UNSPSC) | **220 MB** | **Etapa 1** (Curaduría y saneamiento taxonómico). |
| **`03_oro_benchmark_secop_vec/`** | `secop_vec_corpus_base.parquet` ($1.990.000$ docs), `secop_vec_consultas_10k.parquet` ($10.000$ consultas), `vectores_corpus_e5_768d_f16.mmap` **2,85 GB**, `vectores_consultas_e5_768d_f32.npy` y `verdad_exacta_top100_*.npy` | **3,10GB** | **Etapa 2** (`02_generador_vectores_y_verdad.py` en GPU) e insumos base para las **Etapas 3 a 6** en Mac M1 ($8	ext{ GB}$). |
| **`04_tablas_y_metricas_tesis/`** | `tabla_3_1_estadisticas_corpus.json`, `tabla_3_2_verdad_referencia_gpu.json`, `tabla_4_1_frontera_pareto_p1_p3_2.json`, `tabla_4_3_jerarquia_vs_plano_p2.json`, `tabla_4_4_busqueda_filtrada_p4.json` y `tabla_4_5_validacion_prototipo_rag.json` | **100	KB** | **Respaldo de tablas experimentales** (también versionadas en este repositorio GitHub). |
| **`05_indice_produccion_usearch/`** | `secop_int8_dim512_M12.usearch` (Índice HNSW cuantizado en `int8`, dimensión Matryoshka $512$, conectividad $M=12$) | **1,17 GB** | **Etapa 6** (`06_prototipo_buscador_rag_secop.py`): ejecución inmediata del buscador con `view=True` sin reconstruir el grafo. |

---

## 2. Estructura del Repositorio

    SECOP_Vec_Local/
    ├── README.md                                 # Documentación y mapa de rutas en Google Drive
    ├── Makefile                                  # Orquestación reproducible en un comando
    ├── pyproject.toml                            # Configuración de pytest
    ├── config/
    │   └── experimentos_tesis.yaml               # Hiperparámetros declarativos (Etapas 1 a 6)
    ├── entornos/
    │   ├── requirements_colab_gpu.txt            # Entorno A: Fábrica de datos y GPU (Colab Pro)
    │   └── requirements_mac_m1_8gb.txt           # Entorno B: Laboratorio local de 8 GB RAM
    ├── notebooks_colab/
    │   ├── corpus_secop.ipynb.                   # Cuaderno 1: Primer Módulo Google Colab
    │   └── generador_vectores_y_verdad.ipynb     # Cuaderno 2: Segundo Módulo Google Colab
    ├── tests/
    │   └── test_integridad_cientifica.py         # Suite pytest (invariantes matemáticos y datos)
    ├── 01_fabrica_corpus_secop.py                # Etapa 1: Ingesta Socrata y deduplicación
    ├── 02_generador_vectores_y_verdad.py         # Etapa 2: Embeddings E5 y fuerza bruta GPU
    ├── 03_experimento_cuantizacion.py            # Etapa 3: Frontera de Pareto P1 y H3
    ├── 04_experimento_jerarquia_nsw.py           # Etapa 4: HNSW Jerárquico vs. Grafo Plano P2
    ├── 05_experimento_busqueda_filtrada_p4.py    # Etapa 5: Búsqueda filtrada y ACORN-1 P4
    ├── 06_prototipo_buscador_rag_secop.py        # Etapa 6: Buscador semántico + Capa RAG
    ├── 07_generador_graficas_tesis.py            # Etapa 7: Exportación de figuras PDF/PNG 300 DPI
    ├── 04_tablas_y_metricas_tesis/               # Resultados JSON oficiales
    └── 06_figuras_tesis_ieee/                    # Figuras vectoriales listas para publicación

---

## 3. Guía Rápida de Reproducción para Evaluadores

1. **Preparar el entorno aislado en macOS / Linux (8 GB RAM):**

       python3 -m venv .venv
       make instalar-entorno-mac

2. **Descargar desde Google Drive hacia las carpetas locales:**
   - Descargue el contenido de `03_oro_benchmark_secop_vec/` en `./03_oro_benchmark_secop_vec/`.
   - Descargue `05_indice_produccion_usearch/secop_int8_dim512_M12.usearch` en `./05_indices_temporales_ssd/`.

3. **Ejecutar la auditoría matemática (`pytest`) y regenerar las figuras científicas:**

       make auditar-todo

4. **Ejecutar el prototipo funcional del buscador con síntesis RAG:**

       make etapa6
