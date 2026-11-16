# ==============================================================================
# MAKEFILE DE REPRODUCIBILIDAD CIENTÍFICA — PROYECTO SECOP-VEC (8 GB RAM)
# ==============================================================================
PYTHON := .venv/bin/python

.PHONY: ayuda instalar-entorno-mac test etapa3 etapa4 etapa5 etapa6 figuras auditar-todo

ayuda:
	@echo "Comandos disponibles para reproducir la tesis SECOP-Vec:"
	@echo "  make instalar-entorno-mac : Instala las librerías del entorno de 8 GB"
	@echo "  make test                 : Ejecuta las pruebas unitarias y matemáticas (pytest)"
	@echo "  make etapa3               : Ejecuta Frontera de Pareto (P1 y P3)"
	@echo "  make etapa4               : Ejecuta Jerarquía HNSW vs. Grafo Plano (P2)"
	@echo "  make etapa5               : Ejecuta Búsqueda Filtrada y ACORN-1 (P4)"
	@echo "  make etapa6               : Ejecuta Prototipo Buscador + Capa RAG (Obj. 5)"
	@echo "  make figuras              : Genera las 6 figuras IEEE/ACM en PDF y PNG"
	@echo "  make auditar-todo         : Ejecuta pytest + regenera figuras"

instalar-entorno-mac:
	$(PYTHON) -m pip install -r entornos/requirements_mac_m1_8gb.txt

test:
	$(PYTHON) -m pytest tests/test_integridad_cientifica.py -v

etapa3:
	$(PYTHON) 03_experimento_cuantizacion.py --fase completa

etapa4:
	$(PYTHON) 04_experimento_jerarquia_nsw.py

etapa5:
	$(PYTHON) 05_experimento_busqueda_filtrada_p4.py

etapa6:
	$(PYTHON) 06_prototipo_buscador_rag_secop.py
	$(PYTHON) 06b_parche_casos_rag.py

figuras:
	$(PYTHON) 07_generador_graficas_tesis.py

auditar-todo: figuras test
	@echo "=== AUDITORÍA DE REPRODUCIBILIDAD COMPLETADA CON ÉXITO ==="
