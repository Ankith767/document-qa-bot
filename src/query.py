import os
import sys
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

load_dotenv()

client_genai = genai.Client(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────
# CUSTOM EMBEDDING FUNCTION
# ─────────────────────────────────────────
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using new google-genai SDK."""

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = client_genai.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings


# ─────────────────────────────────────────
# LOAD VECTOR DATABASE
# ─────────────────────────────────────────
def load_collection():
    """Loads the existing ChromaDB collection from disk."""
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = GeminiEmbeddingFunction()

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    return collection


# ─────────────────────────────────────────
# RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────
def retrieve_context(collection, user_query: str) -> tuple[str, list[str]]:
    """Searches ChromaDB for the most relevant chunks to the query."""

    results = collection.query(
        query_texts=[user_query],
        n_results=TOP_K
    )

    context_blocks = []
    citations = []

    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source = meta['source']
        page = meta['page']
        citation = f"{source} (Page {page})"

        context_blocks.append(f"[Source: {source}, Page: {page}]\n{doc}")
        citations.append(citation)

    context_payload = "\n\n---\n\n".join(context_blocks)
    return context_payload, citations


# ─────────────────────────────────────────
# GENERATE ANSWER WITH GEMINI
# ─────────────────────────────────────────
def generate_answer(user_query: str, context_payload: str) -> str:
    """Sends the query + context to Gemini and returns a grounded answer."""

    prompt = f"""You are a professional, accurate document Q&A assistant.
Answer the user's question using ONLY the provided document context below.
Cite the sources (filenames and page numbers) inline next to every fact you mention.
If the answer cannot be found in the context, clearly state:
'I am sorry, but the provided documents do not contain the answer to your question.'
Do not make up facts or use external knowledge.

CONTEXT INFORMATION:
{context_payload}

USER QUESTION: {user_query}

GROUNDED ANSWER:"""

    response = client_genai.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text


# ─────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────
def query_rag_pipeline(user_query: str) -> dict:
    """Full RAG pipeline: retrieve + generate + return answer with citations."""

    collection = load_collection()
    context_payload, citations = retrieve_context(collection, user_query)
    answer = generate_answer(user_query, context_payload)

    return {
        "answer": answer,
        "citations": citations
    }


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 Testing RAG pipeline...\n")
    test_query = "What is the attention mechanism in transformers?"
    result = query_rag_pipeline(test_query)

    print(f"Question: {test_query}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used:")
    for c in result['citations']:
        print(f"  📄 {c}")