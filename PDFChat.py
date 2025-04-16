import streamlit as st
import random,time,os,requests
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
    saved_path = os.path.join(upload_folder, file_name)

    with open(saved_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"PDF file has successfully uploaded to {saved_path}")

    reader = PdfReader(saved_path)
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text()  # Extract text from the first page

    st.write(text)

def response_generator(text,prompt):
    # API_URL = "https://router.huggingface.co/hf-inference/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    # headers = {"Authorization": "Bearer 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"}

    API_URL = "http://127.0.0.1:11434/api/text"

    payload = ({
        "model" : "deepseek-r1:1.5b",
    "inputs": {
        "question": prompt,
        "context": text
    },
    })

    response = requests.post(API_URL, json=payload, stream=True)
    output = response.json()

    return output

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
    
    # Ensure text is not empty before calling response_generator
    if text:
        response = response_generator(text, prompt)
        with st.chat_message("assistant"):
            st.markdown(response['answer'])
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    else:
        st.error("No text extracted from the PDF.")
