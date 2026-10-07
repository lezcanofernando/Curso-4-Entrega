"""Ingesta: lee los documentos de data/, los divide en fragmentos y los sube a Pinecone."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (CATEGORIAS, CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, INDEX_NAME,
                    NAMESPACE, get_embeddings)
from setup_index import crear_indice


def cargar_documentos():
    """Lee todos los archivos .md de la carpeta data/."""
    documentos = []
    for archivo in sorted(Path(DATA_DIR).glob("*.md")):
        texto = archivo.read_text(encoding="utf-8")
        metadata = {"fuente": archivo.name, "categoria": CATEGORIAS.get(archivo.name, "general")}
        documentos.append(Document(page_content=texto, metadata=metadata))
    return documentos


def cargar_fragmentos():
    """Lee los documentos y los divide en fragmentos. También lo usa el recuperador BM25."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    fragmentos = splitter.split_documents(cargar_documentos())

    # Numera los fragmentos dentro de cada archivo: 1, 2, 3...
    contador = {}
    for frag in fragmentos:
        fuente = frag.metadata["fuente"]
        contador[fuente] = contador.get(fuente, 0) + 1
        frag.metadata["fragmento"] = contador[fuente]
    return fragmentos


def ingestar():
    crear_indice()
    fragmentos = cargar_fragmentos()
    print(f"Fragmentos generados: {len(fragmentos)}")

    # Cada fragmento tiene un id fijo (ej. "01_recepcion_mercaderia.md#1"):
    # si se vuelve a correr la ingesta, se sobrescriben en lugar de duplicarse.
    ids = [f"{f.metadata['fuente']}#{f.metadata['fragmento']}" for f in fragmentos]

    # PineconeVectorStore guarda el texto original en la metadata (clave "text"),
    # junto con fuente, categoria y fragmento. Así no hace falta otra base de datos.
    PineconeVectorStore.from_documents(
        fragmentos, get_embeddings(), index_name=INDEX_NAME, namespace=NAMESPACE, ids=ids
    )
    print(f"Ingesta completa: {len(fragmentos)} fragmentos en '{INDEX_NAME}' (namespace '{NAMESPACE}').")


if __name__ == "__main__":
    ingestar()
