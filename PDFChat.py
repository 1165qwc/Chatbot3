import streamlit as st
import os,requests
from huggingface_hub import InferenceClient
from pypdf import PdfReader

upload_folder = 'uploaded_pdf_file'

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

st.header("PDF Chatbot")

uploaded_file = st.file_uploader("Choose a pdf file", type=['pdf','PDF'])

text = ""  # Initialize text variable

if uploaded_file is not None:
    file_name = uploaded_file.name
    saved_path = os.path.join(upload_folder,file_name)

    try:
        with open(saved_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"PDF file has successfully uploaded to {saved_path}")

        reader = PdfReader(saved_path)
        number_of_pages = len(reader.pages)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

        if text:
            st.write(text)
        else:
            st.error("No text could be extracted from the PDF.")
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")

def response_generator(text,prompt):
    API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
    headers = {"Authorization": "Bearer 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"}

    payload = {
        "inputs": {
            "question": prompt,
            "context": text
        },
        "max_tokens": 512,
        "model": "accounts/fireworks/models/deepseek-v3-0324"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        output = response.json()
        return output
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while contacting the API: {e}")
        return None

st.title("Simple Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if text:
        response = response_generator(text, prompt)
        if response:
            with st.chat_message("assistant"):
                st.markdown(response.get('answer', 'No answer found.'))
            st.session_state.messages.append({"role": "assistant", "content": response.get('answer', 'No answer found.')})
    else:
        st.error("No text available to generate a response.")
    
