# Pre-entrega 4: Sistema RAG escalable en la nube con Pinecone

Asistente interno para las sucursales de **Supermercados del Valle**, una cadena de supermercados ficticia. Es la continuación de la pre-entrega 3: la base vectorial local (ChromaDB) se reemplazó por un índice **Pinecone Serverless**, y la búsqueda ahora es **híbrida**: combina búsqueda semántica (vectores) con búsqueda léxica (BM25). También hay un script que mide la calidad de la recuperación con **Precision@5** y **Recall@5**.

## Archivos

| Archivo | Qué hace |
|---|---|
| `data/` | Dataset: 4 documentos `.md` con procedimientos de la empresa. |
| `config.py` | Configuración compartida: nombre del índice, namespace, modelo de embeddings, chunking y `TOP_K`. |
| `setup_index.py` | **Setup**: verifica si el índice existe y, si no, lo crea en modo Serverless. |
| `ingest.py` | **Ingesta**: lee `data/`, divide en fragmentos y los sube a Pinecone con su metadata. |
| `rag_system.py` | **Recuperador híbrido**: la clase `RAGSystem` (BM25 + Pinecone con `EnsembleRetriever`) y la respuesta con Claude. |
| `golden_set.json` | 5 preguntas de prueba con el documento que debería encontrarse para cada una. |
| `evaluate.py` | **Evaluación**: calcula Precision@5 y Recall@5 con el golden set. |
| `demo.py` | Las pruebas de la pre-entrega 3 (respuestas de Claude), ahora con el recuperador híbrido. |
| `schemas.py` | Modelo Pydantic de la respuesta: texto + referencias. |

## Cómo funciona

### 1. Índice en Pinecone (`setup_index.py`)

Crea el índice si no existe, con:

- **Serverless** en `aws / us-east-1` (disponible en el plan gratuito).
- **Dimensión 1024**, la misma que generan los embeddings (ver abajo). Si el índice ya existe con otra dimensión, el script lo avisa con un error en lugar de intentar subir vectores incompatibles.
- **Métrica coseno**.

### 2. Embeddings

Se usa `multilingual-e5-large`, un modelo que **Pinecone aloja y ejecuta** (Pinecone Inference). Ventajas: funciona bien en español, entra en el plan gratuito y no hace falta otra API key (como la de OpenAI): alcanza con la de Pinecone. Genera vectores de **1024 dimensiones**.

La ingesta y la consulta usan la misma función, `get_embeddings()` en `config.py`, así siempre se usa el mismo modelo.

### 3. Ingesta (`ingest.py`)

1. Lee los archivos `.md` de `data/`.
2. Los divide con `RecursiveCharacterTextSplitter` en fragmentos de **1000 caracteres (~250 tokens) con 150 de solapamiento**. Como los documentos son cortos (~2700 caracteres cada uno), cada uno queda en 3 o 4 fragmentos que coinciden más o menos con sus secciones. Con chunks de 500-800 tokens cada documento quedaría en un solo fragmento y la búsqueda no podría distinguir entre secciones.
3. Sube los fragmentos a Pinecone en el **namespace `procedimientos`**, para que estos documentos no se mezclen con otros datos que se guarden en el mismo índice.

Cada vector se guarda con esta metadata:

| Campo | Ejemplo | Para qué sirve |
|---|---|---|
| `text` | "Los productos refrigerados..." | El **texto original** del fragmento. Así no hace falta otra base de datos para recuperarlo. |
| `fuente` | `01_recepcion_mercaderia.md` | El archivo de origen (lo cita Claude y lo usa la evaluación). |
| `categoria` | `logistica` | Etiqueta de categoría del documento. |
| `fragmento` | `3` | Número de fragmento dentro del archivo (equivale a la "página"). |

Cada vector tiene un id fijo (por ejemplo `01_recepcion_mercaderia.md#3`), así que si se corre la ingesta otra vez, los fragmentos se **sobrescriben** en vez de duplicarse.

### 4. Recuperador híbrido (`rag_system.py`)

La clase `RAGSystem` arma dos recuperadores y los combina:

```python
vectorial = PineconeVectorStore(...).as_retriever(search_kwargs={"k": 5})  # semántico
lexico = BM25Retriever.from_documents(cargar_fragmentos(), k=5)            # léxico
retriever = EnsembleRetriever(retrievers=[lexico, vectorial], weights=[0.5, 0.5])
```

- **Vectorial (Pinecone)**: encuentra fragmentos con el mismo *significado* aunque usen otras palabras ("me lastimé" ≈ "accidente de trabajo").
- **BM25**: encuentra fragmentos con las mismas *palabras*. Sirve para términos exactos como "ART", "cash back" o "0800", que los embeddings a veces no distinguen bien. Trabaja en memoria sobre los mismos fragmentos que se subieron a Pinecone.
- **EnsembleRetriever**: une los dos rankings (Reciprocal Rank Fusion) dándole el mismo peso a cada uno.

Métodos:

- `buscar(consulta)`: devuelve los **top-5** fragmentos combinados.
- `responder(pregunta)`: busca los fragmentos y le pide a Claude (`claude-sonnet-4-5`) que responda **solo** con esa información, o "No lo sé." si no está (igual que en la pre-entrega 3).

### 5. Evaluación (`evaluate.py`)

`golden_set.json` tiene 5 preguntas con el formato `{"pregunta": "...", "documento_id_esperado": "..."}`. Para cada pregunta se recuperan los top-5 fragmentos y se calcula:

- **Recall@5**: 1 si algún fragmento del documento esperado aparece entre los 5 recuperados; 0 si no.
- **Precision@5**: qué parte de los 5 fragmentos recuperados viene del documento esperado (de 0 a 1).

Al final se imprime el promedio de las dos métricas.

**Ojo con la precisión**: cada documento tiene solo 3 o 4 fragmentos, así que aunque el recuperador sea perfecto, de los 5 fragmentos al menos 1 o 2 van a ser de otros documentos. La Precision@5 máxima posible es 0,6–0,8, no 1.

## Cómo replicar el índice

Requiere Python 3.12 y dos API keys:

- **Pinecone**: crear una cuenta gratis en <https://app.pinecone.io> y copiar la API key.
- **Anthropic**: solo hace falta para `demo.py` (las respuestas de Claude). La ingesta y la evaluación no la usan.

Los comandos se ejecutan desde la carpeta del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # completar PINECONE_API_KEY, ANTHROPIC_API_KEY e INDEX_NAME

python setup_index.py       # crea el índice Serverless (si no existe)
python ingest.py            # sube los fragmentos a Pinecone
python evaluate.py          # imprime Precision@5 y Recall@5
python demo.py              # preguntas de prueba con respuestas de Claude
```

`ingest.py` también llama a `setup_index.py`, así que se puede saltear el segundo paso. Después de la ingesta, en la consola de Pinecone se ve el índice con 13 vectores en el namespace `procedimientos`.

## Resultados de la evaluación

> Completar con la salida de `python evaluate.py`.

```text
=== Resumen ===
Preguntas evaluadas: 5
Recall@5 promedio:    ...
Precision@5 promedio: ...
```

## Seguridad

Las API keys se leen de `.env`, que está en `.gitignore` y no se sube al repositorio.
