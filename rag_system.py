"""RAGSystem: recuperador híbrido (BM25 + Pinecone) y generación de la respuesta con Claude."""

from langchain_anthropic import ChatAnthropic
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from config import INDEX_NAME, NAMESPACE, TOP_K, get_embeddings
from ingest import cargar_fragmentos
from schemas import RespuestaRAG

parser = PydanticOutputParser(pydantic_object=RespuestaRAG)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres el asistente interno de las sucursales de Supermercados del Valle.\n"
     "Responde SOLO con la información del CONTEXTO. No uses conocimiento propio.\n"
     'Si la respuesta no está en el CONTEXTO, responde exactamente "No lo sé." '
     "y deja la lista de referencias vacía.\n\n"
     "{instrucciones_formato}"),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
]).partial(instrucciones_formato=parser.get_format_instructions())


class RAGSystem:
    def __init__(self, k=TOP_K):
        self.k = k

        # 1. Búsqueda semántica: similitud de vectores en Pinecone.
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME, embedding=get_embeddings(), namespace=NAMESPACE
        )
        vectorial = vectorstore.as_retriever(search_kwargs={"k": k})

        # 2. Búsqueda léxica: BM25 sobre los mismos fragmentos. Encuentra coincidencias exactas
        #    de palabras (nombres, códigos, números) que los embeddings pueden pasar por alto.
        lexico = BM25Retriever.from_documents(cargar_fragmentos(), k=k)

        # 3. EnsembleRetriever: combina los dos rankings, con el mismo peso para cada uno.
        self.retriever = EnsembleRetriever(retrievers=[lexico, vectorial], weights=[0.5, 0.5])

        # Modelo para generar la respuesta (igual que en la pre-entrega 3).
        self.chain = prompt | ChatAnthropic(model="claude-sonnet-4-5", temperature=0, max_tokens=1000) | parser

    def buscar(self, consulta):
        """Devuelve los top-k documentos combinando resultados léxicos y semánticos."""
        # Cada recuperador trae k; el ensemble los une y ordena, y nos quedamos con los k mejores.
        return self.retriever.invoke(consulta)[: self.k]

    def responder(self, pregunta):
        """Busca los documentos y le pide a Claude una respuesta basada solo en ellos."""
        documentos = self.buscar(pregunta)
        contexto = "\n\n".join(f"[fuente: {d.metadata['fuente']}]\n{d.page_content}" for d in documentos)
        return self.chain.invoke({"contexto": contexto, "pregunta": pregunta})
