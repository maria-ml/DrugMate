import streamlit as st
import base64
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import os
from collections import Counter
import time
import torch
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)
from langdetect import detect_langs
from difflib import SequenceMatcher
import re
import pycountry
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')




########## UI SETUP ##############


# Page configuration:
st.set_page_config(page_title="💊 Medical Chatbot", layout="wide")

with open("logo.png", "rb") as f:
    img_data = f.read()
img_base64 = base64.b64encode(img_data).decode()

# Custom CSS:
st.markdown(f"""
    <style>
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: #faf0f2;
        }}
        .image-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 2rem;
        }}
        .title {{
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
            color: #4CAF50;
            margin-top: 1rem;
        }}
        .subtitle {{
            font-size: 1.2rem;
            text-align: center;
            color: #555;
            margin-bottom: 2rem;
        }}
        .stTextInput > div > div > input {{
            font-size: 1.1rem;
        }}
        button {{
            font-size: 1.05rem;
            background-color: white;
            color: black;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            transition: 0.3s;
        }}
        button:hover {{
            background-color: #a8d98f;
            color: black;
            border: 2px solid #45a049;
        }}
        .response-box {{
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.1rem;
            color: #1b5e20;
        }}
        .summary-box {{
            background-color: #eceff1;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.05rem;
            color: #263238;
        }}
        .source-list {{
            background-color: #e3f2fd;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            color: #0d47a1;
        }}
    </style>

    <div class="image-wrapper">
        <img src="data:image/png;base64,{img_base64}" width="200"/>
    </div>
""", unsafe_allow_html=True)

# HEADER:
st.markdown('<div class="title">🤖 Drug Information Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask about a medicine and receive real data from medical leaflets.</div>', unsafe_allow_html=True)
st.markdown("---")

# Session state initialization:
if "response_text" not in st.session_state:
    st.session_state.response_text = None
    st.session_state.pdf_sources = None
    st.session_state.context = None
    st.session_state.user_input = ""
    st.session_state.suggestions = []
    st.session_state.chat_history = []
    st.session_state.summary_text = None
    st.session_state.new_question = False


if "summarize_clicked" not in st.session_state:  
    st.session_state.summarize_clicked = False
if "followup_clicked" not in st.session_state:  
    st.session_state.followup_clicked = None




# -------------------------------------------------------------------------------------------------


########## LLM (LANGUAGE MODEL) INTEGRATION ##############


# ----------------------------- #
# FAISS AND RETRIEVER FUNCTION
# ----------------------------- #

# Load FAISS
@st.cache_resource
def load_faiss():
    """Loads a cached FAISS retriever with MiniLM embeddings for medical document search (top-4 results)."""
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 4})

retriever = load_faiss()


API_KEY = "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"


# ----------------------------- #
# LLM QUERY HANDLING
# ----------------------------- #

def chat_with_llama3(query, max_retries=10, initial_delay=7):
    """
    Queries leaflet documents using FAISS retrieval and generates responses via Llama3.
    Handles dosage-specific queries, drug name disambiguation, and provides sourced answers.
    Implements retry logic for API reliability.
    Returns: (answer, sources, context)
    """  
    results = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in results])

    # Extraer metadata
    sources_info = []
    drug_names = []
    for doc in results:
        source = doc.metadata['source']
        lines = doc.page_content.splitlines()
        section_title = lines[1].strip() if len(lines) >= 2 else "Unknown Section"
        sources_info.append((source, section_title))

        base_name = os.path.basename(source).split("_")[0].lower()
        drug_names.append(base_name)

    counts = Counter(drug_names)
    repeated_drugs = [drug for drug, count in counts.items() if count > 1 and drug in query.lower()]

    # Detectar si la query menciona una dosis específica
    dosage_match = re.search(r"(\d+)\s*(mg|g|mcg|μg|ml|microgram|micrograms)", query.lower())

    if dosage_match:
        number = dosage_match.group(1)  # Solo el número, por ejemplo '500'
        
        patterns = [rf"[_]{number}[\s_]*mg",rf"[_]{number}[\s_]*ml",rf"[_]{number}[_]",rf"[_]{number}[\s_]*microgram",rf"[_]{number}[\s_]*micrograms"]

        matched_doc = next(
            (doc for doc in results if any(re.search(p, doc.metadata["source"].lower()) for p in patterns)),
            None
        )

        if matched_doc:
            results = [matched_doc]
            sources_info = [(matched_doc.metadata["source"], matched_doc.page_content.splitlines()[1].strip())]

        else:
            # No encontrado ➔ mostrar advertencia pero seguir con todos
            return (f"Sorry, I couldn’t find this specific med with {number} dosage. "
                    "If you want to ask me for another dosage or just want to know general information, I will help you"), sources_info, context

    else:
        # Si no se menciona número específico
        specific_words = ["mg", "g", "mcg", "μg","ml","microgram","micrograms"
                          "tablet", "tablets", "capsule", "capsules",
                          "solution", "drops", "injection", "injectable",
                          "cream", "ointment", "spray"]

        query_mentions_specific = any(re.search(rf"\b{x}\b", query.lower()) for x in specific_words)

        if query_mentions_specific:
            # si el usuario ha especificado ➔ quedarse solo con el primero
            results = [results[0]]
            sources_info = [sources_info[0]]

        elif not repeated_drugs:
            # No hay colisión entre las opciones ➔ usar solo el primero
            results = [results[0]]
            sources_info = [sources_info[0]]

        elif repeated_drugs and not query_mentions_specific:
            # Si es una pregunta general (como "generic" o "general"), usar todos los documentos como source
            generic_keywords = ["generic", "general", "generally", "overview", "summary"]

            if any(word in query.lower() for word in generic_keywords):
                pass

            else:
                # Si hay colisión y no especifica la dosis ➔ pedir aclaración (mostrar mensaje)
                repeated = repeated_drugs[0]
                options = sorted(set(
                    os.path.basename(res.metadata['source']).replace(".pdf", "")
                    for res, name in zip(results, drug_names)
                    if name == repeated
                ))
                clarification_message = (
                    f"There are multiple versions of {repeated.capitalize()} available "
                    f"(e.g., {', '.join(options)}). Please specify which one (including dosage in mg) you mean or if you want to know general information about {repeated.capitalize()}."
                )
                return clarification_message, sources_info, context

    # Crear el contexto
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""<s>[INST] You are a helpful medical assistant. 
A person is going to ask you something about a drug or medicine that you will have to answer in a conversational style 
Only use the context below to answer the question. Please answer briefly in no more than 6 lines. If you do not know the answer, say:
"I am sorry but right now I am afraid don't have that information. You are welcome to ask me another question". Do not invent the answer
For example if you do not know anything about a drug or medicine do not invent it and also if I ask you about things not related with medicines do not give that information cause you dont know it.

Context:
{context}

Question: {query}
Answer: [/INST]"""

    # Retry mechanism
    attempt = 0
    delay = initial_delay

    while attempt < max_retries:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300
                },
                timeout=60
            )
            data = response.json()
            if "choices" in data:
                answer = data["choices"][0]["message"]["content"]
                return answer.strip(), sources_info, context
        except Exception:
            pass

        time.sleep(delay)
        attempt += 1

    return "Error: No valid answer returned after retries.", sources_info, context




########## TRANSLATION SYSTEM ##############


# ----------------------------- #
# (1) MODEL SETUP
# ----------------------------- #

@st.cache_resource
def load_translation_model():
    """
    Loads and caches the M2M100 translation model (418M params) with tokenizer.
    Automatically uses GPU if available. 
    Returns: (tokenizer, model, device)
    """
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_translation_model()


# ----------------------------- #
# (2) LANGUAGE DETECTION FUNCTION
# ----------------------------- #

def detect_language_with_confidence(text, threshold=0.85):
    """
    Detects the language of input text with confidence thresholding, with special handling for CJK (Chinese, Japanese, and Korean) characters.
    
    Args:
        text: Input string to analyze
        threshold: Minimum confidence score (0-1) to accept detection (default: 0.85)
    
    Returns:
        str: 2-letter language code (ISO 639-1), with special cases:
             - 'zh' for Chinese (simplified)
             - 'ja' for Japanese
             - 'ko' for Korean
             - 'en' as fallback
    """
    # General detection first
    general_languages = detect_langs(text)
    print("General detection:", general_languages)  # See what languages are generally detected

    print (general_languages)
    if general_languages and general_languages[0].prob >= threshold:
        general_language = general_languages[0].lang
        if general_language =="zh-cn":
            return "zh" 
        elif general_language not in ["zh", "ja", "ko"]:
            return general_language  # Return the detected language (like Spanish, French, etc.)

    # Filter Chinese characters (kanji)
    chinese_text = ''.join([c for c in text if '\u4e00' <= c <= '\u9fff'])

    # Filter Japanese characters (Hiragana + Katakana)
    japanese_text = ''.join([c for c in text if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF'])

    # Filter Korean characters (Hangul)
    korean_text = ''.join([c for c in text if '\uAC00' <= c <= '\uD7AF'])

    # If Japanese text is found (Hiragana or Katakana), prioritize Japanese
    if japanese_text:
        languages = detect_langs(japanese_text)
        print("Japanese detected:", languages)  # See what languages are detected
        if languages and languages[0].prob >= threshold:
            return "ja"  # Return Japanese

    # If Korean text is found (Hangul)
    if korean_text:
        languages = detect_langs(korean_text)
        print("Korean detected:", languages)
        if languages and languages[0].prob >= threshold:
            return "ko"  # Return Korean

    # If Chinese text (kanji only) is found, attempt detection
    if chinese_text:
        languages = detect_langs(chinese_text)
        print("Chinese detected:", languages)  # See what languages are detected
        if languages and languages[0].prob >= threshold:
            return "zh"  # Return Chinese

    # If no Chinese, Japanese, or Korean is detected, return general language
    return "en"  # Default to English if no other language is detected


# ----------------------------- #
# (3) TRANSLATION FUNCTION
# ----------------------------- #

def clean_and_format_text(text):
    """
    Cleans and standardizes text formatting by:
    1. Fixing punctuation spacing (commas, periods)
    2. Normalizing whitespace
    3. Capitalizing sentences properly

    Args:
        text (str): Input text to be cleaned

    Returns:
        str: Formatted text with consistent punctuation and capitalization
    """
    # Corregir espacios y puntuación básica
    text = text.replace("..", ".")
    text = re.sub(r'\s*\.\s*', '. ', text)  
    text = re.sub(r'\s*,\s*', ', ', text)   
    text = re.sub(r'\s+', ' ', text)        

    # Forzar mayúscula tras punto
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.capitalize() for s in sentences]
    return " ".join(sentences)


def postprocess_translation_output(text):
    """
    Cleans translation output by removing unwanted bullet characters or dashes at line starts.
    Handles cases where the translation incorrectly adds list indicators to non-list text.

    Args:
        text (str): Raw translated text possibly containing leading bullets/dashes

    Returns:
        str: Cleaned text with leading bullets/dashes removed from each line
    """
    # Quitar guion largo o viñeta inicial si no forma parte real de una lista
    return re.sub(r"^[–\-•]\s*", "", text.strip(), flags=re.MULTILINE)


def translate_with_format(text, source_language, target_language):
    """
    Translates text while preserving its original formatting structure, with special handling for:
    - Bulleted/numbered lists (maintains list structure)
    - Sentence boundaries (keeps paragraphs separated)
    - Proper punctuation and capitalization

    Args:
        text (str): Text to be translated
        source_language (str): 2-letter source language code (e.g., 'es', 'zh')
        target_language (str): 2-letter target language code (e.g., 'en', 'ja')

    Returns:
        str: Formatted translated text with preserved structure
    """
    # Si el texto parece una lista mal formateada (ej. empieza con "*", "-", etc.)
    list_pattern = r'(?m)^[\*\-•]\s*(.+)'  # Captura ítems de lista por línea

    matches = re.findall(list_pattern, text)
    
    if matches:
        # Buscar encabezado (línea antes del primer ítem de lista)
        header = text.split(matches[0])[0].strip()
        translated_header = ""
        if header:
            translated_header = translate(header, source_language, target_language)
            translated_header = clean_and_format_text(translated_header)

        translated_items = []
        for item in matches:
            translated = translate(item.strip(), source_language, target_language)
            translated = clean_and_format_text(translated)
            translated_items.append(f"* {translated}")

        return f"{translated_header}\n\n" + "\n".join(translated_items)
    
    else:
        # Si no hay listas, traducir por oraciones
        sentences = sent_tokenize(text)
        translated_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            translated = translate(sentence, source_language, target_language)
            translated = clean_and_format_text(translated)
            translated_sentences.append(translated)

        return "\n\n".join(translated_sentences)



def are_similar(text1, text2, threshold=0.95):
    """
    Determines if two texts are nearly identical using sequence matching.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        threshold (float): Similarity score cutoff (0.0 to 1.0). Default 0.95 (95% similar)
        
    Returns:
        bool: True if texts meet the similarity threshold, False otherwise
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() >= threshold



def translate(text, source_language, target_language):
    """
    Translates text between languages using the M2M100 model.

    Args:
        text (str): Text to be translated
        source_language (str): 2-letter ISO code of source language (e.g., 'en', 'es')
        target_language (str): 2-letter ISO code of target language (e.g., 'fr', 'zh')

    Returns:
        str: Translated text with special tokens removed
    """
    with torch.no_grad():
        tokenizer.src_lang = source_language
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_language),
            max_length=512
        )
        return tokenizer.decode(translated[0], skip_special_tokens=True)





########## MULTILINGUAL CHATBOT ##############


def multilingual_chatbot(user_question):
    """
    Handles multilingual medical queries by:
    1. Detecting input language
    2. Translating non-English questions to English
    3. Generating responses using medical knowledge base
    4. Translating responses back to original language

    Args:
        user_question (str): Medical question in any supported language

    Returns:
        tuple: (response_text, pdf_sources, context, detected_language_code)
    """
    user_language = detect_language_with_confidence(user_question)
    #print(user_question)

    # Translate introductory phrase

    if user_language == "en":
        response, pdfs, context = chat_with_llama3(user_question)
        #print(response)
        return response, pdfs, context, user_language
    else:
        #print("Detected non-English language")
        lang_source = user_language
        lang_target = "en"

        question_in_english = translate(user_question, lang_source, lang_target)
        #print("Question in English:", question_in_english)

        if lang_source == "en" or are_similar(user_question, question_in_english):
            #print("⚡ The question was actually already in English. Correcting language...")
            response, pdfs, context = chat_with_llama3(user_question)
            return response, pdfs, context

        language_name = pycountry.languages.get(alpha_2=user_language).name
        intro_phrase = f"Oh! It seems you want to speak in {language_name}. I'll answer in this language too."
        translated_intro_phrase = translate_with_format(intro_phrase, "en", user_language)
        st.write(f"{translated_intro_phrase}")
        #print(question_in_english)
        # Get chatbot response in English
        response_in_english, pdfs, context = chat_with_llama3(question_in_english)
        #print("Response in English:", response_in_english)

        # Translate the response
        translated_response = translate_with_format(response_in_english, lang_target, lang_source)
        translated_response = postprocess_translation_output(translated_response)

        # Combine both parts
        final_response = f"{translated_response}"
        return final_response, pdfs, context, user_language







# -------------------------------------------------------------------------------------------------

########## UI INTERACTION ##############


# ----------------------------- #
# CONVERSATION HISTORY
# ----------------------------- #

# TOGGLE BUTTON: Show/Hide history
show_history = st.toggle("🕑 Show Conversation History")

# SHOW history immediately if toggled
if show_history:
    st.markdown("## 🕑 Full Conversation History")
    if len(st.session_state.chat_history) >= 1:
        for idx, (question, answer) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{idx}:** {question}")
            st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("No conversation yet. Ask your first question!")

# Divider
st.markdown("---")


# ----------------------------- #
# INPUT A QUESTION
# ----------------------------- #


# INPUT: New question
with st.form("question_form", clear_on_submit=False):
    input_box = st.text_input("💬 Type your question:", value=st.session_state.user_input, key="input_box")
    send = st.form_submit_button("🚀 Send Question")

if send and input_box.strip() != "":
    st.session_state.user_input = input_box
    st.session_state.new_question = True
    st.session_state.summary_text = None  
    st.rerun()  


# HANDLE new question
if st.session_state.new_question and st.session_state.user_input.strip() != "":
    start_time = time.time() 
    with st.spinner("Generating response..."):
        try:
            response_text, pdf_sources, context, user_language = multilingual_chatbot(st.session_state.user_input)
            st.session_state.response_text = response_text
            st.session_state.pdf_sources = pdf_sources
            st.session_state.context = context
            st.session_state.user_language = user_language
            st.session_state.chat_history.append((st.session_state.user_input, response_text))  
            st.session_state.summary_text = None


            # ----------------------------- #
            # FOLLOW-UP QUESTIONS 
            # ----------------------------- #

            # Generate follow-up questions
            suggestion_prompt = f"""Based on the user's question:
"{st.session_state.user_input}"

Suggest 3 short follow-up questions the user might ask next. 
Always include the name of the drug in the question, so that the chat can have context.
ALWAYS generate the bullet list in the same language as the following question: "{st.session_state.user_language}"
Return only the questions as a bullet list starting with "• ".
No introductions or explanations.
"""
            response_suggestion = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": suggestion_prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300
                },
                timeout=30
            )

            response_json_suggestion = response_suggestion.json()
            if "choices" in response_json_suggestion:
                suggestions = response_json_suggestion["choices"][0]["message"]["content"]
                suggestion_lines = [line.strip() for line in suggestions.split("•") if line.strip()]
                st.session_state.suggestions = suggestion_lines[-3:]
            else:
                st.session_state.suggestions = []

        except Exception as e:
            st.error(f"An error occurred: {e}")

    end_time = time.time()  # Tiempo de finalización

    # Calcular el tiempo total
    elapsed_time = end_time - start_time
    st.session_state.response_time = elapsed_time  # Guardar el tiempo de respuesta en la sesión

    # Mostrar el tiempo de respuesta
    st.write(f"⏱️Generated response in: {elapsed_time:.2f} seconds.")
    
    st.session_state.new_question = False


# DISPLAY: Last question and answer
if len(st.session_state.chat_history) >= 1:
    last_question, last_answer = st.session_state.chat_history[-1]
    st.markdown("## ✨ Latest Question and Answer")
    st.markdown(f"**You asked:** {last_question}")
    st.markdown(f'<div class="response-box">{last_answer}</div>', unsafe_allow_html=True)

# Divider
st.markdown("---")




# FOLLOW-UP + SUMMARIZE
if len(st.session_state.chat_history) >= 1:
    if st.session_state.suggestions:
        st.markdown("### 🤔 Suggested Follow-up Questions:")
        for idx, suggestion in enumerate(st.session_state.suggestions):
            if st.button(f"🟠 {suggestion}", key=f"followup_{idx}"):
                st.session_state.user_input = suggestion
                st.session_state.new_question = True
                st.session_state.summary_text = None
                st.rerun()  


    # ----------------------------- #
    # SUMMARIZATION OF CONTEXT
    # ----------------------------- #

    if st.button("🔎 Summarize context"):
        with st.spinner("Summarizing context..."):
            if st.session_state.context:
                summary_prompt = f"""Summarize the following medical information in a few sentences for a non-expert user. 
                            
                                    No yapping. No introductions. Directly output only the clean summary without any extra sentences. 
                                    Do it short, so it's easy to read for the user.

{st.session_state.context}

"""
                try:
                    response_summary = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": "Bearer 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",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama3-8b-8192",
                            "messages": [{"role": "user", "content": summary_prompt}],
                            "temperature": 0.3,
                            "max_tokens": 300
                        },
                        timeout=30
                    )
                    response_json_summary = response_summary.json()
                    if "choices" in response_json_summary:
                        summary = response_json_summary["choices"][0]["message"]["content"]
                        if st.session_state.user_language != 'en':
                            summary = translate_with_format(summary, 'en', st.session_state.user_language) 
                        st.session_state.summary_text = summary
                    else:
                        st.warning("⚠️ Could not generate a summary.")
                except Exception as e:
                    st.warning(f"⚠️ Error while summarizing: {e}")
            else:
                st.warning("⚠️ No context available to summarize.")


    if st.session_state.summary_text:
        st.markdown("### 📝 Sources Summary:")
        st.markdown(f'<div class="summary-box">{st.session_state.summary_text}</div>', unsafe_allow_html=True)

    if st.session_state.pdf_sources:
        st.markdown("### 📄 Source Documents:")
        for source in st.session_state.pdf_sources:
            st.markdown(f'<div class="source-list">{source}</div>', unsafe_allow_html=True)


# Divider
st.markdown("---")

# MEET THE TEAM Section
st.markdown("## 👥 Meet the Team")

# Toggle Show/Hide
show_team = st.toggle("📸 Show Our Team")

# Mostrar imagen y nombres si el toggle está activado
if show_team:

    with open("team_photo.jpg", "rb") as f:
        img_data = f.read()
    img_base64 = base64.b64encode(img_data).decode()

    st.markdown(f"""
        <style>
            .team-wrapper {{
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-top: 3rem;
                margin-bottom: 2rem;
            }}
            .team-caption {{
                font-style: italic;
                color: #444;
                margin-top: 0.5rem;
            }}
            .team-members {{
                text-align: center;
                font-size: 1.1rem;
                margin-top: 1.5rem;
                line-height: 1.6;
            }}
        </style>

        <div class="team-wrapper">
            <img src="data:image/jpeg;base64,{img_base64}" width="400"/>
            <div class="team-caption">Our Amazing Team</div>
            <div class="team-members">
                <strong>Team Members:</strong><br>
                Paula de Diego<br>
                María Martínez<br>
                Paula Ochotorena<br>
                Camilo Zabala
            </div>
        </div>
    """, unsafe_allow_html=True)

