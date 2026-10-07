"""Pruebas: una pregunta con respuesta en los documentos y dos preguntas trampa."""

from rag_system import RAGSystem

preguntas = [
    # 1. La respuesta está en 01_recepcion_mercaderia.md.
    "¿Qué hago si llega un camión de lácteos a más de 8 °C?",
    # 2. Trampa: el documento de personal existe, pero no dice nada de sueldos.
    "¿Cuánto cobra un repositor por mes?",
    # 3. Trampa: no tiene nada que ver con los documentos.
    "¿Quién ganó el Mundial de fútbol de 2022?",
]

rag = RAGSystem()
for pregunta in preguntas:
    print(f"\nPregunta: {pregunta}")
    print(rag.responder(pregunta).model_dump_json(indent=2))
