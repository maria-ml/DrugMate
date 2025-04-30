# DrugMate – AI-Powered Drug Information Assistant

**DrugMate** is a multilingual, AI-powered chatbot designed to provide users with **reliable, document-based answers** about medications. It uses cutting-edge **natural language processing** techniques and a custom-trained document database of official drug leaflets from the [Health Products Regulatory Authority (HPRA)](https://www.hpra.ie/). 

Users can ask questions about medication usage, side effects, dosages, and contraindications, and receive grounded, trustworthy responses.

---

## Key Features of the System

- **Document-grounded answers** – no hallucinations
- **Multilingual support** – 100+ languages via automatic detection and translation
- **Semantic search** with FAISS and MiniLM
- **Conversational responses** powered by LLaMA 3 (Groq)
- **Follow-up question generation** to keep users engaged
- **Summarization of complex leaflet content**
- **Streamlit UI** for a clean and responsive interface

---

## Technologies & Architecture employed

| Component                               | Role                                |
|----------------------------------------|-------------------------------------|
| `sentence-transformers/all-MiniLM-L12-v2` | Embedding model for semantic search |
| `FAISS`                                 | Vector database for fast retrieval  |
| `llama3-8b-8192` via Groq API           | Large Language Model (LLM)          |
| `facebook/m2m100_418M`                  | Translation model for multilingual support |
| `langdetect`                            | Language detection                  |
| `Streamlit`                             | Web application UI                  |

---








