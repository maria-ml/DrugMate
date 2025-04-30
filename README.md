# DrugMate – AI-Powered Drug Information Assistant

**DrugMate** is a multilingual, AI-powered chatbot designed to provide users with **reliable, document-based answers** about medications. It uses cutting-edge **natural language processing** techniques and a custom-trained document database of official drug leaflets from the [Health Products Regulatory Authority (HPRA)](https://www.hpra.ie/). 

Users can ask questions about medication usage, side effects, dosages, and contraindications, and receive grounded, trustworthy responses.

<div align="center">
<img src="https://github.com/user-attachments/assets/ba5e635f-12b1-4841-955f-642d4c71621e" width="300">
</div>


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


## Getting Started

### Prerequisites
TODO

### Installation
```bash
# Clone the repository
git clone https://github.com/TODO
cd TODO

# Install dependencies
pip install -r requirements.txt TODO

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env  # If using OpenAI TODO
```
---
## How to Use DrugMate TODO: meter capturas

1) Launch the application
2) Ask medication questions in natural language. Try these type of questions:
TODO
3) Receive accurate, sourced answers

