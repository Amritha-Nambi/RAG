# 🧙‍♂️ Hogwarts AI Librarian: Advanced RAG Chatbot

**A full-stack Retrieval-Augmented Generation (RAG) application that answers complex questions about the Harry Potter universe with high accuracy.**

This project moves beyond simple vector search by implementing **Hybrid Search (Vector + Keyword)**, **Cross-Encoder Reranking**, and **Context Expansion** to synthesize answers from scattered narrative fragments.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Stack](https://img.shields.io/badge/Tech-FAISS%20%7C%20LangChain%20%7C%20Streamlit%20%7C%20Gemini-green)

---

## 🚀 Key Features

* **Hybrid Search Architecture:** Combines semantic understanding (**FAISS/Vector Search**) with precise keyword matching (**BM25**) to solve the "needle in a haystack" problem.
* **Smart Reranking:** Uses a Cross-Encoder (`ms-marco-MiniLM`) to deeply analyze and re-score retrieved chunks for maximum relevance.
* **Context Expansion:** Automatically retrieves neighboring text chunks to provide the LLM with full scene context, not just isolated sentences.
* **Modular "Orchestrator" Pattern:** Clean architecture separating data pipelines, core logic, and frontend code.
* **LLM-as-a-Judge Evaluation:** Includes an automated evaluation pipeline where Gemini grades the chatbot's accuracy against a Golden Dataset.
* **Streamlit UI:** A modern, interactive web interface for chatting with the system.

---

## 📂 Project Structure

```text
harry_potter_rag/
│
├── 📄 main.py                   # 🕹️ THE COMMANDER: Main entry point for all actions
├── 📄 app.py                    # 💻 THE FRONTEND: Streamlit web interface
├── 📄 requirements.txt          # dependencies
├── 📄 .env                      # API keys (not committed)
│
├── 📁 data/                     # 💾 THE VAULT
│   ├── 📁 pdfs/                 # Raw PDF files
│   ├── 📁 extracted_text/       # Intermediate extraction
│   ├── 📁 cleaned_text/         # Cleaned for processing
│   ├── 📄 chunks.pkl            # Text chunks and metadata
│   ├── 📄 embeddings.pkl        # Raw embeddings
│   └── 📄 vector_store.index    # FAISS Vector Index
│
└── 📁 src/                      # 🧠 THE BRAIN
    ├── 📄 pipeline.py           # ETL Pipeline (Extract, Clean, Chunk, Embed)
    ├── 📄 chatbot.py            # RAG Logic (Hybrid Search & Reranking)
    ├── 📄 data_processing.py    # PDF Extraction logic
    ├── 📄 chunking.py           # Recursive Character Splitter logic
    ├── 📄 embedding.py          # SentenceTransformer logic
    └── 📄 utils.py              # Helper functions