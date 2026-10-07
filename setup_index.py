"""Inicialización: verifica si el índice de Pinecone existe y, si no, lo crea (Serverless)."""

from pinecone import Pinecone, ServerlessSpec

from config import DIMENSION, INDEX_NAME


def crear_indice():
    pc = Pinecone()  # toma PINECONE_API_KEY del entorno

    if pc.has_index(INDEX_NAME):
        dimension = pc.describe_index(INDEX_NAME).dimension
        print(f"El índice '{INDEX_NAME}' ya existe (dimensión {dimension}).")
        # Evita el error de subir vectores de una dimensión a un índice de otra.
        if dimension != DIMENSION:
            raise ValueError(f"El índice tiene dimensión {dimension} y los embeddings {DIMENSION}. "
                             "Bórralo desde la consola de Pinecone o usa otro INDEX_NAME.")
        return

    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,  # debe coincidir con el modelo de embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # región del plan gratuito
    )
    print(f"Índice '{INDEX_NAME}' creado (Serverless, dimensión {DIMENSION}, coseno).")


if __name__ == "__main__":
    crear_indice()
