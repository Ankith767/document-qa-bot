
import os
import sys
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai as genai_client
from pypdf import PdfReader
from docx import Document
from tqdm import tqdm
from dotenv import load_dotenv


# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

load_dotenv()

# ─────────────────────────────────────────
# STEP 1: EXTRACT TEXT FROM PDF
# ─────────────────────────────────────────
def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extracts text page by page from a PDF file."""
    extracted_data = []
    file_name = os.path.basename(file_path)

    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1
                    }
                })
    except Exception as e:
        print(f"❌ Error reading PDF {file_name}: {e}")

    return extracted_data


# ─────────────────────────────────────────
# STEP 2: EXTRACT TEXT FROM DOCX
# ─────────────────────────────────────────
def extract_docx_pages(file_path: str) -> list[dict]:
    """Extracts text from a Word document."""
    file_name = os.path.basename(file_path)

    try:
        doc = Document(file_path)
        full_text = "\n".join([
            para.text for para in doc.paragraphs if para.text.strip()
        ])

        return [{
            "text": full_text,
            "metadata": {
                "source": file_name,
                "page": 1
            }
        }]
    except Exception as e:
        print(f"❌ Error reading DOCX {file_name}: {e}")
        return []


# ─────────────────────────────────────────
# STEP 3: CHUNK THE EXTRACTED TEXT
# ─────────────────────────────────────────
def chunk_extracted_pages(pages: list[dict]) -> list[dict]:
    """Splits page text into smaller overlapping chunks."""
    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + CHUNK_SIZE, text_length)
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}"
                }
            })

            start += (CHUNK_SIZE - CHUNK_OVERLAP)

    return chunks


# ─────────────────────────────────────────
# STEP 4: SAVE TO VECTOR DATABASE
# ─────────────────────────────────────────
def save_to_vector_db(chunks: list[dict]):
    """Embeds chunks and saves them to ChromaDB."""
    client = chromadb.PersistentClient(path=DB_PATH)

    _client = genai_client.Client(api_key=GEMINI_API_KEY)

    class GeminiEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            embeddings = []
            for text in input:
                result = _client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text
                )
                embeddings.append(result.embeddings[0].values)
            return embeddings

    embedding_fn = GeminiEmbeddingFunction()

    # Delete existing collection to avoid duplicate ID errors
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("🗑️  Cleared existing collection.")
    except:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"id_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Upload in batches of 50
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size), desc="📥 Indexing chunks"):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"\n✅ Successfully indexed {len(chunks)} chunks into the vector database!")


# ─────────────────────────────────────────
# MAIN: RUN THE FULL INGESTION PIPELINE
# ─────────────────────────────────────────
if __name__ == "__main__":
    all_pages = []

    print("\n📂 Scanning documents in data/ folder...\n")

    files = os.listdir(DATA_PATH)
    if not files:
        print("❌ No files found in data/ folder. Add some PDFs or DOCXs first!")
        sys.exit(1)

    for file in tqdm(files, desc="📄 Extracting text"):
        file_path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            pages = extract_pdf_pages(file_path)
            all_pages.extend(pages)
            print(f"  ✅ PDF extracted: {file} ({len(pages)} pages)")

        elif file.endswith(".docx"):
            pages = extract_docx_pages(file_path)
            all_pages.extend(pages)
            print(f"  ✅ DOCX extracted: {file}")

        else:
            print(f"  ⚠️  Skipped unsupported file: {file}")

    if not all_pages:
        print("❌ No text could be extracted. Check your documents.")
        sys.exit(1)

    print(f"\n📊 Total pages extracted: {len(all_pages)}")

    chunks = chunk_extracted_pages(all_pages)
    print(f"✂️  Total chunks created: {len(chunks)}")

    save_to_vector_db(chunks)