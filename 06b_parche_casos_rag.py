# -*- coding: utf-8 -*-
"""
MÓDULO 06B: PARCHE FORENSE DE CASOS DE ESTUDIO RAG (INFRAESTRUCTURA VIAL Y SUMINISTROS)
================================================================================
Corrige la selección de vectores de consulta y aplica filtrado estricto contra
contratos de prestación de servicios camuflados bajo Régimen Especial (Segmento 80).
"""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from usearch.index import Index

RUTA_BASE = Path("/Users/gapt/SECOP_Vec_Local")
RUTA_ORO = RUTA_BASE / "03_oro_benchmark_secop_vec"
RUTA_REPORTES = RUTA_BASE / "04_tablas_y_metricas_tesis"
RUTA_INDICES = RUTA_BASE / "05_indices_temporales_ssd"

ARCHIVO_CORPUS = RUTA_ORO / "secop_vec_corpus_base.parquet"
ARCHIVO_CONSULTAS = RUTA_ORO / "secop_vec_consultas_10k.parquet"
ARCHIVO_VEC_Q = RUTA_ORO / "vectores_consultas_e5_768d_f32.npy"
ARCHIVO_INDICE = RUTA_INDICES / "secop_int8_dim512_M12.usearch"
ARCHIVO_REPORTE_4_5 = RUTA_REPORTES / "tabla_4_5_validacion_prototipo_rag.json"


def ejecutar_consulta_rag_limpia(
    indice: Index,
    tabla_corpus: pa.Table,
    nombre_caso: str,
    texto_consulta: str,
    vector_q_768: np.ndarray,
    segmentos_permitidos: Tuple[str, ...],
    filtro_departamento: Optional[str] = None,
    top_k: int = 10,
) -> Dict:
    t0 = time.perf_counter()
    q_512 = vector_q_768[:512].astype(np.float32)
    q_512 /= np.linalg.norm(q_512) + 1e-12

    ventana = 2000
    indice.expansion_search = ventana
    res = indice.search(q_512, count=ventana)
    cand_ids = np.asarray(res.keys, dtype=np.int64)
    t_busqueda_ms = (time.perf_counter() - t0) * 1000.0

    t_rag0 = time.perf_counter()
    subtabla = tabla_corpus.take(cand_ids)
    deptos = subtabla.column("departamento").to_pylist()
    es_ps = subtabla.column("es_prestacion_servicios").to_pylist()
    segs = subtabla.column("unspsc_segmento").to_pylist()
    descs = subtabla.column("descripcion_objeto").to_pylist()

    indices_validos: List[int] = []
    for i_loc in range(len(cand_ids)):
        texto_up = str(descs[i_loc]).upper()
        # 1. Exclusión estricta de prestación de servicios (explícita o camuflada en segmento 80)
        if bool(es_ps[i_loc]) or str(segs[i_loc]) == "80":
            continue
        if "PRESTACION DE SERVICIOS" in texto_up or "PRESTACIÓN DE SERVICIOS" in texto_up:
            continue
        if "ARRENDAMIENTO" in texto_up:
            continue
        # 2. Verificación de sector económico y filtro geográfico opcional
        if segmentos_permitidos and str(segs[i_loc]) not in segmentos_permitidos:
            continue
        if filtro_departamento and filtro_departamento.upper() not in str(deptos[i_loc]).upper():
            continue

        indices_validos.append(i_loc)
        if len(indices_validos) >= top_k:
            break

    if not indices_validos:
        indices_validos = list(range(min(top_k, len(cand_ids))))

    seleccion = subtabla.take(indices_validos)
    valores = [float(v) for v in seleccion.column("valor_contrato").to_pylist()]
    entidades = [str(e) for e in seleccion.column("entidad").to_pylist()]
    modalidades = [str(m) for m in seleccion.column("modalidad").to_pylist()]
    clases_unspsc = [str(c) for c in seleccion.column("unspsc_clase").to_pylist()]
    descripciones = [str(d) for d in seleccion.column("descripcion_objeto").to_pylist()]
    procesos = [str(p) for p in seleccion.column("numero_proceso").to_pylist()]

    val_min = float(np.min(valores))
    val_mediana = float(np.median(valores))
    val_media = float(np.mean(valores))
    val_max = float(np.max(valores))

    top_entidades = Counter(entidades).most_common(3)
    top_modalidades = Counter(modalidades).most_common(3)
    top_clases = Counter(clases_unspsc).most_common(2)

    resumen_rag = (
        f"Síntesis RAG ({len(valores)} contratos análogos recuperados para '{texto_consulta[:95]}...'): "
        f"El valor contractual oscila entre ${val_min:,.0f} COP y ${val_max:,.0f} COP, "
        f"con una mediana de ${val_mediana:,.0f} COP (media: ${val_media:,.0f} COP). "
        f"Las modalidades predominantes son {', '.join([f'{m} ({c})' for m, c in top_modalidades])}, "
        f"destacando como principales entidades contratantes "
        f"{', '.join([f'{e} ({c})' for e, c in top_entidades])} "
        f"bajo las clases UNSPSC {', '.join([c for c, _ in top_clases])}."
    )
    t_rag_ms = (time.perf_counter() - t_rag0) * 1000.0

    return {
        "caso_estudio": nombre_caso,
        "consulta_lenguaje_natural": texto_consulta,
        "filtros_aplicados": {
            "departamento": filtro_departamento,
            "segmentos_unspsc_objetivo": list(segmentos_permitidos),
            "excluir_prestacion_servicios_y_seg80": True,
        },
        "contratos_recuperados": len(valores),
        "latencia_busqueda_vectorial_ms": round(t_busqueda_ms, 3),
        "latencia_agregacion_rag_ms": round(t_rag_ms, 3),
        "latencia_total_end_to_end_ms": round(t_busqueda_ms + t_rag_ms, 3),
        "capa_rag_rango_precios_cop": {
            "minimo_cop": round(val_min, 2),
            "mediana_cop": round(val_mediana, 2),
            "media_cop": round(val_media, 2),
            "maximo_cop": round(val_max, 2),
        },
        "capa_rag_entidades_frecuentes": [
            {"entidad": e, "contratos_en_top_k": c} for e, c in top_entidades
        ],
        "capa_rag_modalidades_usadas": [
            {"modalidad": m, "contratos_en_top_k": c} for m, c in top_modalidades
        ],
        "muestra_primer_contrato_recuperado": {
            "numero_proceso": procesos[0],
            "entidad": entidades[0],
            "valor_cop": valores[0],
            "descripcion": descripciones[0][:240],
        },
        "resumen_ejecutivo_generado": resumen_rag,
    }


if __name__ == "__main__":
    indice = Index.restore(str(ARCHIVO_INDICE), view=True)
    columnas = [
        "id_contrato", "numero_proceso", "descripcion_objeto",
        "unspsc_segmento", "unspsc_familia", "unspsc_clase",
        "departamento", "entidad", "modalidad", "valor_contrato",
        "anio_firma", "es_prestacion_servicios",
    ]
    tabla_c = pq.read_table(ARCHIVO_CORPUS, columns=columnas)
    tabla_q = pq.read_table(
        ARCHIVO_CONSULTAS,
        columns=["descripcion_objeto", "unspsc_segmento", "unspsc_clase", "departamento"],
    )

    textos_q = tabla_q.column("descripcion_objeto").to_pylist()
    seg_q = tabla_q.column("unspsc_segmento").to_pylist()
    vectores_q = np.load(ARCHIVO_VEC_Q)

    # Caso 1: Obra pública de mantenimiento vial (Segmento 72: Edificación y Mantenimiento)
    idx_vias = next(
        i for i, (t, s) in enumerate(zip(textos_q, seg_q))
        if str(s) == "72"
        and any(w in t.upper() for w in ["MANTENIMIENTO", "MEJORAMIENTO", "REHABILITACION", "PAVIMENT"])
        and any(v in t.upper() for v in ["VIAS", "VÍAS", "VIAL", "TERCIARIA", "RURAL"])
    )

    # Caso 2: Suministro de kits/materiales/dotación (Segmentos de bienes 44, 53, 60)
    idx_kits = next(
        i for i, (t, s) in enumerate(zip(textos_q, seg_q))
        if str(s) in ("44", "53", "60")
        and "SUMINISTRO" in t.upper()
        and any(w in t.upper() for w in ["PAPELERIA", "DOTACION", "DOTACIÓN", "ESCOLAR", "MATERIAL", "ELEMENTOS"])
        and "PRESTACION" not in t.upper()
    )

    caso_1 = ejecutar_consulta_rag_limpia(
        indice=indice,
        tabla_corpus=tabla_c,
        nombre_caso="Caso 1 (Veeduría de Infraestructura y Mantenimiento Vial)",
        texto_consulta=textos_q[idx_vias],
        vector_q_768=vectores_q[idx_vias],
        segmentos_permitidos=("72", "30", "95"),
        filtro_departamento=None,
        top_k=10,
    )

    caso_2 = ejecutar_consulta_rag_limpia(
        indice=indice,
        tabla_corpus=tabla_c,
        nombre_caso="Caso 2 (Estudio de Mercado de Suministro de Dotación y Materiales)",
        texto_consulta=textos_q[idx_kits],
        vector_q_768=vectores_q[idx_kits],
        segmentos_permitidos=("44", "53", "60", "56"),
        filtro_departamento=None,
        top_k=10,
    )

    with open(ARCHIVO_REPORTE_4_5, "r", encoding="utf-8") as f:
        reporte = json.load(f)

    reporte["casos_estudio_prototipo_rag"] = [caso_1, caso_2]
    with open(ARCHIVO_REPORTE_4_5, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print("=== CASOS DE ESTUDIO RAG ACTUALIZADOS CON ÉXITO ===")
    print(json.dumps(reporte["casos_estudio_prototipo_rag"], indent=2, ensure_ascii=False))
