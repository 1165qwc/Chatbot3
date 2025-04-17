# Import necessary libraries
import streamlit as st
import random
import time
import os
import requests
from huggingface_hub import InferenceClient
from pypdf import PdfReader

# Define the folder for storing uploaded PDF files
upload_folder = 'uploaded_pdf_file'

# Create the folder if it doesn't exist
if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

# Streamlit header for the app
st.header("PDF Chatbot")

# File uploader for PDF files
uploaded_file = st.file_uploader("Choose a pdf file", type=['pdf', 'PDF'])

# If a file is uploaded, save and process it
if uploaded_file is not None:
    file_name = uploaded_file.name
    saved_path = os.path.join(upload_folder, file_name)

    # Save the uploaded file to the specified folder
    with open(saved_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display success message once the file is uploaded
    st.success(f"PDF file has successfully uploaded to {saved_path}")

    # Extract text from the first page of the PDF
    reader = PdfReader(saved_path)
    number_of_pages = len(reader.pages)
    page = reader.pages[0]  # Extract text from the first page
    text = page.extract_text()

    # Display extracted text
    st.write(text)

# Function to generate a response from Hugging Face model
def response_generator(text, prompt):
    """
    This function uses Hugging Face's API to get the response from a pretrained model.
    It sends the input context (PDF text) and a prompt (question) to the model.
    """
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

# Streamlit title for the chatbot
st.title("Simple Chat")

# Initialize session state to store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages (if any)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for the chat
if prompt := st.chat_input("Ask something related to the PDF text"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save the user's message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response using the response_generator function
    response = response_generator(text, prompt)
    
    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    
    # Save the assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})

