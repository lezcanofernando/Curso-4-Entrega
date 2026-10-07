"""Configuración compartida por todos los scripts."""

import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings

load_dotenv()  # lee PINECONE_API_KEY, ANTHROPIC_API_KEY e INDEX_NAME del archivo .env

DATA_DIR = "data"
INDEX_NAME = os.getenv("INDEX_NAME", "supermercado-docs")
NAMESPACE = "procedimientos"  # separa estos documentos de otros datos del mismo índice

# Modelo de embeddings alojado en Pinecone: multilingüe y sin necesidad de otra API key.
# Genera vectores de 1024 dimensiones, así que el índice se crea con esa dimensión.
EMBEDDING_MODEL = "multilingual-e5-large"
DIMENSION = 1024

CHUNK_SIZE = 1000    # caracteres por fragmento (~250 tokens)
CHUNK_OVERLAP = 150  # caracteres que se repiten entre un fragmento y el siguiente
TOP_K = 5            # documentos que devuelve el recuperador

# Etiqueta de categoría de cada documento (se guarda en la metadata).
CATEGORIAS = {
    "01_recepcion_mercaderia.md": "logistica",
    "02_reposicion_y_stock.md": "stock",
    "03_cajas_y_atencion.md": "atencion_cliente",
    "04_personal_y_seguridad.md": "recursos_humanos",
}


def get_embeddings():
    # La ingesta y la consulta usan esta misma función, así siempre se usa el mismo modelo.
    return PineconeEmbeddings(model=EMBEDDING_MODEL)
