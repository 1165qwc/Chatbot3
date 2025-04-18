import streamlit as st
import random, time, os, requests
from huggingface_hub import InferenceClient
from pypdf import PdfReader
import re
from fuzzywuzzy import fuzz
import base64
from io import BytesIO

upload_folder = 'uploaded_pdf_file'

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

st.header("PDF Chatbot")

uploaded_file = st.file_uploader("Choose a pdf file", type=['pdf','PDF'])

extracted_text = ""
if uploaded_file is not None:
    file_name = uploaded_file.name
    saved_path = os.path.join(upload_folder,file_name)

    with open(saved_path,'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"PDF file has successfully uploaded to {saved_path}")

    reader = PdfReader(saved_path)
    number_of_pages = len(reader.pages)
    
    # Extract text from all pages
    extracted_text = ""
    for i in range(number_of_pages):
        page = reader.pages[i]
        extracted_text += page.extract_text()
    
    # Display first page text as preview
    st.write(reader.pages[0].extract_text())

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

def get_pdf_image(pdf_path):
    """Extract first page as image for multimodal models"""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        if images:
            buffered = BytesIO()
            images[0].save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        return None
    except Exception as e:
        st.error(f"Error extracting PDF image: {e}")
        return None

def response_generator(text, prompt, pdf_path=None):
    # Find relevant context and check for missing information
    context, missing_info = fuzzy_match_query(text, prompt)
    
    # Ask for more info if needed
    if missing_info and len(missing_info) > 0:
        return {
            "answer": f"I'd like to help, but I need more information about the {', '.join(missing_info)} to give you a proper answer. Could you please provide more details?",
            "needs_info": True
        }
    
    # Try the advanced Fireworks AI API first (for better responses)
    try:
        fireworks_response = call_fireworks_api(prompt, context, pdf_path)
        if fireworks_response:
            return {"answer": enhance_response(fireworks_response)}
    except Exception as e:
        st.warning(f"Falling back to BERT model: {e}", icon="⚠️")
    
    # Fallback to the original BERT model if Fireworks API fails
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

def call_fireworks_api(prompt, context, pdf_path=None):
    """Call the Fireworks AI API with text and optional image"""
    API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",  # Using same token for now
    }
    
    # Prepare message content
    message_content = [
        {
            "type": "text",
            "text": f"Context from PDF: {context}\n\nUser question: {prompt}\n\nAnswer the question based on the provided context."
        }
    ]
    
    # Add image if available
    if pdf_path:
        image_url = get_pdf_image(pdf_path)
        if image_url:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            )
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ],
        "max_tokens": 512,
        "model": "accounts/fireworks/models/llama4-scout-instruct-basic"
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API error: {response.status_code}")

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
    
    # Pass the PDF path for potential multimodal features
    pdf_path = os.path.join(upload_folder, uploaded_file.name) if uploaded_file else None
    response = response_generator(extracted_text if extracted_text else "", prompt, pdf_path)
    
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
