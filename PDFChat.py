import streamlit as st
import os
from pypdf import PdfReader

# Define the folder to store uploaded PDF files
UPLOAD_FOLDER = 'uploaded_pdf_file'

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# Display the header for the application
st.header("PDF Chatbot")

# File uploader for PDF files
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf', 'PDF'])

# Initialize text variable
text = ""

# Process the uploaded file
if uploaded_file is not None:
    file_name = uploaded_file.name
    saved_path = os.path.join(UPLOAD_FOLDER, file_name)

    # Save the uploaded file
    with open(saved_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"PDF file has been successfully uploaded to {saved_path}")

    # Read the PDF file
    reader = PdfReader(saved_path)
    number_of_pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Display the extracted text
    st.write("PDF content loaded successfully.")

# Function to generate a response based on the PDF content
def response_generator(text, prompt):
    # Simple keyword-based search
    if prompt.lower() in text.lower():
        return f"I found the following information related to '{prompt}':\n\n{text}"
    else:
        return "I'm sorry, I couldn't find any information related to your query in the PDF."

# Display the title for the chat interface
st.title("Simple Chat")

# Initialize session state for messages
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

    # Generate a response if text is available
    if text:
        response = response_generator(text, prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("No text extracted from the PDF.")
