# SECOP-Vec: Búsqueda Semántica a Gran Escala con Memoria Restringida (8 GB RAM)

Repositorio oficial de reproducibilidad científica de la Tesis de Maestría:
**Búsqueda semántica a gran escala con memoria restringida: HNSW cuantizado aplicado a la contratación pública colombiana (SECOP II) en un computador de 8 GB de RAM**.

## 1. Arquitectura Experimental de Dos Entornos

1. **Entorno A — Fábrica de Datos y Verdad de Referencia (Google Colab Pro GPU):**
   - Manifiesto: `entornos/requirements_colab_gpu.txt`
   - Módulos: `notebooks_colab/01_fabrica_corpus_secop.py` y `notebooks_colab/02_generador_vectores_y_verdad.py`
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
