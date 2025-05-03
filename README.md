# DrugMate – AI-Powered Drug Information Assistant

**DrugMate** is a multilingual, AI-powered chatbot designed to provide users with **reliable, document-based answers** about medications. It uses cutting-edge **natural language processing** techniques and a custom-trained document database of official drug leaflets from the [Health Products Regulatory Authority (HPRA)](https://www.hpra.ie/). 

Users can ask questions about medication usage, side effects, dosages, and contraindications, and receive grounded, trustworthy responses.

<div align="center">
<img src="https://github.com/user-attachments/assets/5abd7149-bf58-4632-80f3-6952ec995f8f" width="300">
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
```
DrugMate/
│
├── 📁 faiss_index/ # FAISS index and metadata for semantic search
│ ├── index.faiss # Binary FAISS vector index
│ └── index.pkl ** # Pickle file with ID-to-metadata mapping
│
├── 📁 utils/ # core technologies and evaluation metrics
│ ├── drugMate_core.ipynb # includes PDF sectioning, Vector store creation using FAISS, Question-answering pipeline, Language detection, automatic translation and evaluation.
│ ├── Evaluation_questions.xlsx # questions & ground truth to obtain evaluation metrics
│ └── Evaluation_metrics.xlsx # metrics for the optimization of the chatbot
│
├── 📄 app.py # Main application script (entry point)
├── 🖼️ logo.png # Project logo (used in app UI)
├── 🖼️ team_photo.jpg # Project contributors photo
└── 📄 README.md # Project documentation (this file)
```
**```index.pkl``` is too heavy to be uploaded. Please, download it from this [link](https://drive.google.com/drive/folders/1PUVhNXhk_UoTqBRCxZoki1S49hRMpmvL?usp=sharing).


---
## Getting Started

### Prerequisites
* Python 3.7 or higher
* Windows, Linux, or macOS
* Conda (via Anaconda or Miniconda)

### Installation
```bash
# Clone the repository
git clone https://github.com/maria-ml/DrugMate
cd DrugMate

# Install dependencies
pip install -r requirements.txt TODO
```
### Important
The API key publicly shared in ```app.py``` has an expiration date. After it expires, please create a new API in [Groq](https://console.groq.com/home​).

---
## User Interface Overview

DrugMate provides an interactive web interface that allows users to easily engage with the chatbot and retrieve medical information. Below is a breakdown of the main sections of the interface:

<div align="center">
<img src="https://github.com/user-attachments/assets/24abd5c0-0d5e-4f02-a3f4-cfd8506b490d">
</div>



#### ⏱️ Conversation History

- It allows users to **show or hide full question history**.
- When enabled, users can see:
  - **All previous questions** they've asked in the current session.
  - The **corresponding AI-generated answers** for each.
- This feature helps users revisit prior information without needing to retype their queries.

#### 💬 Question Input

- A text input field allows users to **type questions** about medications in any language.
- A **“Send Question”** button submits the query.
  
#### ⏱️ Response Time Display

- Displays how many seconds the response generation took — a great transparency feature.

#### ✨ Latest Question and Answer

- Shows the **most recent question** and a concise, AI-generated **answer**, sourced strictly from verified documents.
- The response box is styled in green for clarity and trust.

#### 🤔 Suggested Follow-up Questions

- The app automatically generates **3 contextually relevant follow-up questions**.
- Users can click any of them to dive deeper into the topic.
- These are generated in the same language as the original query.

#### 🔍 Summarize Context

- It summarizes the underlying document **context** in simple, layman-friendly terms.
- Useful for users who want to understand the leaflet content without reading full medical documents.

#### 📄 Source Documents

- Displays the **exact source documents and section titles** used to generate the answer.
- This ensures transparency and allows users to verify the medical information.

#### 👥 Meet the Team

- Shows a **team photo** and names of the developers of the tool.


---
## How to Use DrugMate

**1) Launch the application**

**2) Ask medication questions. Try these type of questions:**
  ##### English Examples:
  - What are the main reasons me, as a patient, should not take Amlodipine Fair-Med 5 mg?
  - How should Amlodipine Thame 10 mg be stored after opening?
  - How is Ampres solution administered, and what is the typical dose for adults?
  - What happens if I forget to take my Androcur 100 mg?
  - Could Aripil 5 mg affect my ability to drive or use machines?
  - Is it safe to take Valoid 50 mg if I’ve had too much to drink?
  - Why should I avoid grapefruit juice when taking Budenofalk 9 mg granules?
  - Can Bufomix Easyhaler 320 micrograms be used during an asthma attack?
  - What common side effects should I expect with Cataflam 50 mg?
  - Can I use Dalacin 2 vaginal cream if I'm pregnant?
  - Are there any age restrictions for using Echinaforce Cold & Flu drops?
  - Can Epanutin Infatabs 50 mg affect how other medicines work?

  ##### Multilingual Examples:
  - 🇫🇷 *Quelle est la manière recommandée de prendre Roaccutane 20 mg ?*
  - 🇩🇪 *Zu welcher Tageszeit sollte Simvastatin Teva 10 mg am besten eingenommen werden?*
  - 🇮🇹 *Quanto spesso posso prendere Rowalief 500 mg al giorno?*
  - 🇪🇸 *¿El Ibuprofeno 400mg reduce la inflamación?*
  - 🇵🇹 *Eu tenho um problema cardíaco sério. É arriscado tomar Tadalafil Clonmel 20 mg?*
  - 🇨🇳 加比特利 5 毫克会引起情绪变化或情绪副作用吗

Feel free to ask your own questions — the assistant understands context, dosage, risks, interactions, and more!

**3) Receive accurate and sourced answers**


---

## Team

<div align="center">
<img src="https://github.com/user-attachments/assets/40d4bb27-c0bc-4061-b50d-5a8fe55357ca" width="300">
</div>

Developed by: 
- Paula De Diego Hidalgo
- María Martínez Labrador
- Paula Ochotorena Santacilia
- Camilo Zabala Hurtado

University Carlos III of Madrid (UC3M)
Natural Language Processing – Academic Year 2024/2025





