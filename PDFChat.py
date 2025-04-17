import streamlit as st
import random, time, os, requests
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from datetime import datetime

# ---------------- GREETING LOGIC ----------------
def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning!"
    elif hour < 18:
        return "Good afternoon!"
    else:
        return "Good evening!"

# ---------------- FILE UPLOAD SETUP ----------------
upload_folder = 'uploaded_pdf_file'
if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

# ---------------- UI HEADER & GREETING ----------------
st.header("PDF Chatbot")
greeting = get_greeting()
st.markdown(f"### {greeting} Welcome to your PDF Chatbot Assistant!")

# ---------------- PDF UPLOAD & TEXT EXTRACTION ----------------
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf', 'PDF'])

text = ""
if uploaded_file is not None:
    file_name = uploaded_file.name
    saved_path = os.path.join(upload_folder, file_name)

    with open(saved_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"PDF file has successfully uploaded to {saved_path}")

    reader = PdfReader(saved_path)
    number_of_pages = len(reader.pages)
    text = reader.pages[0].extract_text()

    st.subheader("Extracted Text from Page 1:")
    st.write(text)

# ---------------- HUGGING FACE QA MODEL FUNCTION ----------------
def response_generator(text, prompt):
    API_URL = "https://router.huggingface.co/hf-inference/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    headers = {"Authorization": "Bearer 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"}

    payload = {
        "inputs": {
            "question": prompt,
            "context": text
        },
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    output = response.json()
    return output

# ---------------- CHAT INTERFACE ----------------
st.title("Simple Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate model response
    if text:
        with st.spinner("Thinking..."):
            response = response_generator(text, prompt)
            answer = response.get('answer', 'Sorry, I could not find an answer.')
    else:
        answer = "Please upload a PDF file first."

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
