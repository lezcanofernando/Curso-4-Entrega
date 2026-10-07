"""Evaluación del recuperador híbrido con Precision@k y Recall@k sobre el golden set."""

import json

from config import TOP_K
from rag_system import RAGSystem

with open("golden_set.json", encoding="utf-8") as f:
    golden_set = json.load(f)

rag = RAGSystem(k=TOP_K)
precisiones, recalls = [], []

for caso in golden_set:
    esperado = caso["documento_id_esperado"]
    fuentes = [d.metadata["fuente"] for d in rag.buscar(caso["pregunta"])]

    # Recall@k: 1 si el documento esperado aparece entre los k recuperados, 0 si no.
    recall = 1 if esperado in fuentes else 0
    # Precision@k: qué parte de los k recuperados viene del documento esperado.
    precision = fuentes.count(esperado) / TOP_K

    recalls.append(recall)
    precisiones.append(precision)
    print(f"\nPregunta: {caso['pregunta']}")
    print(f"  Esperado:    {esperado}")
    print(f"  Recuperados: {fuentes}")
    print(f"  Recall@{TOP_K} = {recall}   Precision@{TOP_K} = {precision:.2f}")

print("\n=== Resumen ===")
print(f"Preguntas evaluadas: {len(golden_set)}")
print(f"Recall@{TOP_K} promedio:    {sum(recalls) / len(recalls):.2f}")
print(f"Precision@{TOP_K} promedio: {sum(precisiones) / len(precisiones):.2f}")
