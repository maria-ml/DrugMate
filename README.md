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
## Project Structure
TODO

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
## User Interface Overview

DrugMate provides an interactive web interface that allows users to easily engage with the chatbot and retrieve medical information. Below is a breakdown of the main sections of the interface:

<div align="center">
<img src="https://github.com/user-attachments/assets/742ec922-a1ec-43c7-a28a-fc4b2e383763">
</div>



### ⏱️ Conversation History

- It allows users to **show or hide full question history**.
- When enabled, users can see:
  - **All previous questions** they've asked in the current session.
  - The **corresponding AI-generated answers** for each.
- This feature helps users revisit prior information without needing to retype their queries.

### 💬 Question Input

- A text input field allows users to **type questions** about medications in any language.
- A **“Send Question”** button submits the query.
  
### ⏱️ Response Time Display

- Displays how many seconds the response generation took — a great transparency feature.

### ✨ Latest Question and Answer

- Shows the **most recent question** and a concise, AI-generated **answer**, sourced strictly from verified documents.
- The response box is styled in green for clarity and trust.

### 🤔 Suggested Follow-up Questions

- The app automatically generates **3 contextually relevant follow-up questions**.
- Users can click any of them to dive deeper into the topic.
- These are generated in the same language as the original query.

### 🔍 Summarize Context

- it summarizes the underlying document **context** in simple, layman-friendly terms.
- Useful for users who want to understand the leaflet content without reading full medical documents.

### 📄 Source Documents

- Displays the **exact source documents and section titles** used to generate the answer.
- This ensures transparency and allows users to verify the medical information.

### 👥 Meet the Team

- Shows a **team photo** and names of the developers of the tool.


---
## How to Use DrugMate

1) Launch the application
2) Ask medication questions in natural language. Try these type of questions: TODO
      - ..
      - ..
      - ..
      - ..
      - ..
      - ..
      - ..
      - ..
      - ..
      - ..
      - ..
      - ..
      - ..
    
4) Receive accurate, sourced answers


---

## Team
Developed by: 
- Paula De Diego Hidalgo
- María Martínez Labrador
- Paula Ochotorena Santacilia
- Camilo Zabala Hurtado

University Carlos III of Madrid (UC3M)
Natural Language Processing – Academic Year 2024/2025
