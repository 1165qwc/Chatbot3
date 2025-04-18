import streamlit as st
import random, time, os, requests
from huggingface_hub import InferenceClient
from pypdf import PdfReader
import re
from fuzzywuzzy import fuzz

upload_folder = 'uploaded_pdf_file'

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

st.header("PDF Chatbot")

uploaded_file = st.file_uploader("Choose a pdf file", type=['pdf','PDF'])

if uploaded_file is not None:
    file_name = uploaded_file.name
    saved_path = os.path.join(upload_folder,file_name)

    with open(saved_path,'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"PDF file has successfully uploaded to {saved_path}")

    reader = PdfReader(saved_path)
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text()

    st.write(text)    

def enhance_response(answer):
    # Add more human-like expressions to responses
    greetings = ["I found this for you: ", "Here's what I discovered: ", 
                "Based on the document: ", "According to the PDF: "]
    
    # Add follow-up questions if the answer seems short
    follow_ups = ["\n\nIs there anything specific you'd like to know more about?",
                 "\n\nCan I help you with anything else?",
                 "\n\nWould you like more details on this topic?"]
    
    if len(answer) < 50:  # For short answers
        return f"{random.choice(greetings)}{answer}{random.choice(follow_ups)}"
    else:
        return f"{random.choice(greetings)}{answer}"

def fuzzy_match_query(text, query):
    # Extract key terms from user query
    key_terms = re.findall(r'\b\w+\b', query.lower())
    key_terms = [term for term in key_terms if len(term) > 3]  # Filter out short words
    
    # Find relevant context with fuzzy matching
    paragraphs = text.split('\n\n')
    best_paragraph = ""
    highest_score = 0
    
    for paragraph in paragraphs:
        if len(paragraph.strip()) < 10:
            continue
        
        score = 0
        for term in key_terms:
            # Check for fuzzy matches in paragraph
            for word in re.findall(r'\b\w+\b', paragraph.lower()):
                if len(word) > 3:
                    match_score = fuzz.ratio(term, word)
                    if match_score > 70:  # Threshold for considering it a match
                        score += match_score
        
        if score > highest_score:
            highest_score = score
            best_paragraph = paragraph
    
    # If no good paragraph found, use the entire text
    context = best_paragraph if highest_score > 0 else text
    
    # If we need more info from user
    missing_info = []
    if "when" in query.lower() and not re.search(r'\b(date|day|month|year|time)\b', query, re.IGNORECASE):
        missing_info.append("time period")
    if "where" in query.lower() and not re.search(r'\b(location|place|address|city)\b', query, re.IGNORECASE):
        missing_info.append("location")
        
    return context, missing_info

def response_generator(text, prompt):
    # Find relevant context and check for missing information
    context, missing_info = fuzzy_match_query(text, prompt)
    
    # Ask for more info if needed
    if missing_info and len(missing_info) > 0:
        return {
            "answer": f"I'd like to help, but I need more information about the {', '.join(missing_info)} to give you a proper answer. Could you please provide more details?",
            "needs_info": True
        }
    
    API_URL = "https://router.huggingface.co/hf-inference/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    headers = {"Authorization": "Bearer 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"}

    payload = ({
    "inputs": {
        "question": prompt,
        "context": context
    },
    })

    response = requests.post(API_URL, headers=headers, json=payload)
    output = response.json()
    
    # Enhance the response to make it more human-like
    if 'answer' in output:
        output['answer'] = enhance_response(output['answer'])
    
    return output

st.title("PDF Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your document..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = response_generator(text, prompt)
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
