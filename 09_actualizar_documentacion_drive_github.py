# -*- coding: utf-8 -*-
"""
MÓDULO 09: INTEGRACIÓN DE ARQUITECTURA GOOGLE DRIVE EN GITHUB Y DESCARGADOR
================================================================================
Actualiza `README.md`, `config/experimentos_tesis.yaml` y `Makefile` para
documentar la jerarquía completa en Google Drive (Bronce -> Plata -> Oro ->
Índice .usearch) y permitir la descarga reproducible en equipos de 8 GB.
"""

from pathlib import Path

RUTA_RAIZ = Path("/Users/gapt/SECOP_Vec_Local")


def actualizar_archivos() -> None:
    # 1. Actualizar config/experimentos_tesis.yaml con el mapa completo de Google Drive
    ruta_yaml = RUTA_RAIZ / "config" / "experimentos_tesis.yaml"
    contenido_yaml = ruta_yaml.read_text(encoding="utf-8")

    bloque_drive = """
almacenamiento_persistente_google_drive:
  carpeta_raiz_drive: "Mi unidad/SECOP_Vec_Tesis/"
  enlace_publico_evaluadores: "https://drive.google.com/drive/folders/REEMPLAZAR_CON_TU_ID_DE_CARPETA"
  capas_medallon:
    capa_01_bronce:
      ruta: "01_bronce_socrata_parquet/"
      descripcion: "5.714.490 registros crudos del SECOP II (datos.gov.co jbjy-vk9h) en bloques Parquet Zstd"
    capa_02_plata:
      ruta: "02_plata_corpus_deduplicado/"
      descripcion: "2.000.000 de contratos únicos normalizados y etiquetados jerárquicamente por código UNSPSC"
    capa_03_oro:
      ruta: "03_oro_benchmark_secop_vec/"
      archivos:
        - "secop_vec_corpus_base.parquet (1.990.000 documentos del corpus, ~215 MB)"
        - "secop_vec_consultas_10k.parquet (10.000 consultas estratificadas, ~1.5 MB)"
        - "vectores_corpus_e5_768d_f16.mmap (matriz float16 memmap, 2.85 GB)"
        - "vectores_consultas_e5_768d_f32.npy (consultas exactas float32, 29.3 MB)"
        - "verdad_exacta_top100_indices.npy (ground truth GPU top-100, 3.8 MB)"
        - "verdad_exacta_top100_similitud.npy (similitudes coseno exactas, 3.8 MB)"
    capa_04_reportes:
      ruta: "04_tablas_y_metricas_tesis/"
      descripcion: "Tablas experimentales en JSON auditables (Tablas 3.1, 3.2, 4.1, 4.3, 4.4 y 4.5)"
    capa_05_indice_produccion:
      ruta: "05_indice_produccion_usearch/secop_int8_dim512_M12.usearch"
      tamano_mb: 1193.23
      descripcion: "Índice vectorial ganador de la Frontera de Pareto (int8, 512D, M=12) listo para view=True"
"""
    if "almacenamiento_persistente_google_drive:" not in contenido_yaml:
        ruta_yaml.write_text(contenido_yaml.rstrip() + "\n" + bloque_drive, encoding="utf-8")
        print("[OK] config/experimentos_tesis.yaml actualizado con las 5 capas de Google Drive.")

    # 2. Actualizar README.md con la tabla detallada de rutas de Google Drive (Bronce -> Oro -> .usearch)
    readme_completo = """# SECOP-Vec: Búsqueda Semántica a Gran Escala con Memoria Restringida (8 GB RAM)

Repositorio oficial de reproducibilidad científica de la Tesis de Maestría:
**«Búsqueda semántica a gran escala con memoria restringida: HNSW cuantizado aplicado a la contratación pública colombiana (SECOP II) en un computador de 8 GB de RAM»**.

---

## 1. Arquitectura Híbrida de Almacenamiento (GitHub + Google Drive)

Debido a las restricciones de tamaño de archivos binarios en GitHub (`>100 MB`), este repositorio versiona el **100 % del código fuente, configuraciones declarativas (`YAML`), suite de pruebas (`pytest`), métricas (`JSON`) y figuras vectoriales**, mientras que los artefactos masivos del benchmark `SECOP-Vec v1` y el índice preconstruido `.usearch` residen de forma persistente en **Google Drive** bajo una arquitectura Medallón:

* **Enlace público al Lago de Datos (Solo Lectura para Evaluadores):** `https://drive.google.com/drive/folders/REEMPLAZAR_CON_TU_ID_DE_CARPETA`

| Capa en Google Drive (`Mi unidad/SECOP_Vec_Tesis/`) | Artefactos Almacenados | Volumen | Etapa / Uso Reproducible |
| :--- | :--- | :---: | :--- |
| **`01_bronce_socrata_parquet/`** | `secop_bronce_lote_*.parquet` y `estado_descarga_checkpoint.json` ($5.714.490$ contratos crudos desde `datos.gov.co` recurso `jbjy-vk9h`) | $\sim 1,50\text{ GB}$ | **Etapa 1** (`01_fabrica_corpus_secop.py` en Colab Pro). |
| **`02_plata_corpus_deduplicado/`** | `secop_plata_deduplicado.parquet` ($2.000.000$ de objetos contractuales únicos deduplicados por hash MD5 y clasificados en UNSPSC) | $\sim 220\text{ MB}$ | **Etapa 1** (Curaduría y saneamiento taxonómico). |
| **`03_oro_benchmark_secop_vec/`** | `secop_vec_corpus_base.parquet` ($1.990.000$ docs), `secop_vec_consultas_10k.parquet` ($10.000$ consultas), `vectores_corpus_e5_768d_f16.mmap` ($2,85\text{ GB}$), `vectores_consultas_e5_768d_f32.npy` y `verdad_exacta_top100_*.npy` | **$3,10\text{ GB}$** | **Etapa 2** (`02_generador_vectores_y_verdad.py` en GPU) e insumos base para las **Etapas 3 a 6** en Mac M1 ($8\text{ GB}$). |
| **`04_tablas_y_metricas_tesis/`** | `tabla_3_1_estadisticas_corpus.json`, `tabla_3_2_verdad_referencia_gpu.json`, `tabla_4_1_frontera_pareto_p1_p3_2.json`, `tabla_4_3_jerarquia_vs_plano_p2.json`, `tabla_4_4_busqueda_filtrada_p4.json` y `tabla_4_5_validacion_prototipo_rag.json` | $< 100\text{ KB}$ | **Respaldo de tablas experimentales** (también versionadas en este repositorio GitHub). |
| **`05_indice_produccion_usearch/`** | `secop_int8_dim512_M12.usearch` (Índice HNSW cuantizado en `int8`, dimensión Matryoshka $512$, conectividad $M=12$) | **$1,17\text{ GB}$** | **Etapa 6** (`06_prototipo_buscador_rag_secop.py`): ejecución inmediata del buscador con `view=True` sin reconstruir el grafo. |

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
"""
    (RUTA_RAIZ / "README.md").write_text(readme_completo.strip() + "\n", encoding="utf-8")
    print("[OK] README.md actualizado con la tabla completa de Google Drive (Bronce -> Oro -> .usearch).")


if __name__ == "__main__":
    actualizar_archivos()
