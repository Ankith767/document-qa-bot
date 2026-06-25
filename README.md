# 📄 Document Q&A Bot (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions from your own documents with citations.

## 🚀 Tech Stack
- **Google Gemini** — Embeddings + Answer Generation
- **ChromaDB** — Vector Database
- **Streamlit** — Web UI
- **pypdf + python-docx** — Document Parsing

## ⚙️ Setup

1. Clone the repo:
```bash
   git clone https://github.com/Ankith767/document-qa-bot.git
   cd document-qa-bot
```

2. Create virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Add your API key in `.env`:
GEMINI_API_KEY=your_key_here

5. Add your documents to `data/` folder

## 🏃 Running the App

**Step 1 — Ingest documents:**
```bash
python src/ingest.py
```

**Step 2 — Launch the UI:**
```bash
streamlit run src/main.py
```

## 📁 Project Structure

document-qa-bot/

├── .env                  # API keys (not committed)

├── requirements.txt      # Dependencies

├── data/                 # Your source documents

├── db/                   # ChromaDB vector store

└── src/

├── config.py         # Configuration

├── ingest.py         # Document ingestion pipeline

├── query.py          # RAG query pipeline

└── main.py           # Streamlit UI

## 💡 How It Works
1. Documents are parsed and split into chunks
2. Each chunk is embedded using Gemini `gemini-embedding-001`
3. User query is embedded and matched against chunks
4. Top matching chunks are sent to Gemini with the query
5. Gemini returns a grounded answer with citations


Live/Deploy Link :
https://document-app-bot-ticgrdzf2h7jfy2vsq4xx4.streamlit.app/
