import streamlit as st
import random, time, os, requests
from huggingface_hub import InferenceClient
from pypdf import PdfReader
import re
from fuzzywuzzy import fuzz
import base64
from io import BytesIO
import json
import logging

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PDFChat')

# ----------- Upload Folder Setup -----------
upload_folder = 'uploaded_pdf_file'

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

# ----------- Streamlit UI: Header and Sidebar -----------
st.header("PDF Chatbot")

# Sidebar for model selection
st.sidebar.title("Model Settings")
selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    ["Local Ollama", "Fireworks AI", "OpenRouter", "BERT (Fallback)"],
    index=0
)

# ----------- Ollama Connection Check -----------
if selected_model == "Local Ollama":
    try:
        response = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
        if response.status_code == 200:
            st.sidebar.success(f"✅ Ollama connected: {response.json().get('version', 'unknown version')}")
            ollama_models_available = True
        else:
            st.sidebar.error("❌ Ollama server responded with an error. Status code: " + str(response.status_code))
            ollama_models_available = False
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"❌ Could not connect to Ollama server: {str(e)}")
        st.sidebar.info("Make sure Ollama is running with 'ollama serve' command")
        ollama_models_available = False

    # Ollama model selection
    ollama_model = st.sidebar.selectbox(
        "Choose Ollama Model",
        ["deepseek-r1:1.5b"],
        index=0
    )
    
    # Add a refresh button to test connection again
    if st.sidebar.button("Test Ollama Connection"):
        try:
            response = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
            if response.status_code == 200:
                st.sidebar.success(f"✅ Ollama connected: {response.json().get('version', 'unknown version')}")
                try:
                    model_response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
                    if model_response.status_code == 200:
                        models = model_response.json().get('models', [])
                        model_names = [model.get('name') for model in models]
                        if ollama_model in model_names:
                            st.sidebar.success(f"✅ Model '{ollama_model}' is available!")
                        else:
                            st.sidebar.warning(f"⚠️ Model '{ollama_model}' not found. Available models: {', '.join(model_names)}")
                            st.sidebar.info(f"Try: ollama pull {ollama_model}")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not check model availability: {str(e)}")
            else:
                st.sidebar.error("❌ Ollama server responded with an error")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"❌ Could not connect to Ollama server: {str(e)}")
            st.sidebar.info("Make sure Ollama is running with 'ollama serve' command")

# ----------- OpenRouter API Integration -----------
def call_openrouter_api(prompt, context, pdf_path=None):
    """Call the OpenRouter API for deepseek model"""
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    logger.info("Calling OpenRouter API with deepseek model")
    
    # Prepare message with strict instructions to only use PDF content
    messages = [
        {"role": "system", "content": f"You are a PDF assistant that ONLY answers questions based on the content of the uploaded PDF document. DO NOT use any external knowledge. If the answer cannot be found in the PDF, say 'I cannot find that information in the PDF.' Here is the relevant context from the PDF: {context}"},
        {"role": "user", "content": f"Based ONLY on the PDF content provided, answer this question: {prompt}"}
    ]
    
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": messages
    }
    
    try:
        logger.info(f"Sending request to OpenRouter API")
        response = requests.post(
            url=API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=60
        )
        logger.info(f"OpenRouter API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info("Received successful response from OpenRouter")
            return data["choices"][0]["message"]["content"]
        else:
            error_msg = f"OpenRouter API error: {response.status_code}"
            logger.error(error_msg)
            if response.text:
                logger.error(f"Response text: {response.text}")
            st.error(error_msg)
            return f"Error: {response.status_code}"
    except requests.exceptions.ConnectTimeout:
        error_msg = "Connection timeout when connecting to OpenRouter API."
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to OpenRouter API."
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error calling OpenRouter: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg

# ----------- PDF Upload and Text Extraction -----------
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

# ----------- Response Enhancement Function -----------
def enhance_response(answer):
    greetings = ["I found this for you: ", "Here's what I discovered: ", 
                "Based on the document: ", "According to the PDF: "]
    follow_ups = ["\n\nIs there anything specific you'd like to know more about?",
                 "\n\nCan I help you with anything else?",
                 "\n\nWould you like more details on this topic?"]
    if len(answer) < 50:
        return f"{random.choice(greetings)}{answer}{random.choice(follow_ups)}"
    else:
        return f"{random.choice(greetings)}{answer}"

# ----------- Fuzzy Matching Logic for Context Extraction -----------
def fuzzy_match_query(text, query):
    key_terms = re.findall(r'\b\w+\b', query.lower())
    key_terms = [term for term in key_terms if len(term) > 3]
    
    paragraphs = text.split('\n\n')
    best_paragraph = ""
    highest_score = 0
    
    for paragraph in paragraphs:
        if len(paragraph.strip()) < 10:
            continue
        score = 0
        for term in key_terms:
            for word in re.findall(r'\b\w+\b', paragraph.lower()):
                if len(word) > 3:
                    match_score = fuzz.ratio(term, word)
                    if match_score > 70:
                        score += match_score
        if score > highest_score:
            highest_score = score
            best_paragraph = paragraph
    
    context = best_paragraph if highest_score > 0 else text
    missing_info = []
    if "when" in query.lower() and not re.search(r'\b(date|day|month|year|time)\b', query, re.IGNORECASE):
        missing_info.append("time period")
    if "where" in query.lower() and not re.search(r'\b(location|place|address|city)\b', query, re.IGNORECASE):
        missing_info.append("location")
        
    return context, missing_info

# ----------- PDF Image Extraction for Multimodal Use -----------
def get_pdf_image(pdf_path):
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

# ----------- Ollama API Communication Logic -----------
def call_ollama_api(prompt, context, model="deepseek-r1:1.5b", pdf_path=None):
    """Call the local Ollama API running at 127.0.0.1:11434"""
    API_URL = "http://127.0.0.1:11434/api/chat"
    
    logger.info(f"Calling Ollama API with model: {model}")
    
    # Prepare message for text-based models with strict instructions to only use PDF content
    messages = [
        {"role": "system", "content": f"You are a PDF assistant that ONLY answers questions based on the content of the uploaded PDF document. DO NOT use any external knowledge. If the answer cannot be found in the PDF, say 'I cannot find that information in the PDF.' Here is the relevant context from the PDF: {context}"},
        {"role": "user", "content": f"Based ONLY on the PDF content provided, answer this question: {prompt}"}
    ]
    
    # For multimodal models like llava
    if model.lower() == "llava" and pdf_path:
        image_data = None
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            if images:
                buffered = BytesIO()
                images[0].save(buffered, format="JPEG")
                image_data = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Could not convert PDF to image: {e}")
            st.warning(f"Could not convert PDF to image for multimodal model: {e}")
        
        if image_data:
            # Format for llava multimodal with strict instructions
            API_URL = "http://127.0.0.1:11434/api/generate"
            payload = {
                "model": model,
                "prompt": f"<image>\nYou are a PDF assistant that ONLY answers questions based on the content of the uploaded PDF document. DO NOT use any external knowledge. If the answer cannot be found in the PDF, say 'I cannot find that information in the PDF.'\n\nContext from PDF: {context}\n\nQuestion: {prompt}\n\nAnswer based ONLY on the PDF content:",
                "stream": False,
                "images": [image_data]
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
    else:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
    
    try:
        logger.info(f"Sending request to Ollama API at {API_URL}")
        response = requests.post(API_URL, json=payload, timeout=60)
        logger.info(f"Ollama API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info("Received successful response from Ollama")
            
            # Handle different API response formats
            if "message" in data:
                return data["message"]["content"]
            elif "response" in data:
                return data["response"]
            else:
                logger.warning("Unexpected response format from Ollama")
                return "No response from Ollama"
        else:
            error_msg = f"Ollama API error: {response.status_code}"
            logger.error(error_msg)
            if response.text:
                logger.error(f"Response text: {response.text}")
            st.error(error_msg)
            return f"Error: {response.status_code}"
    except requests.exceptions.ConnectTimeout:
        error_msg = "Connection timeout when connecting to Ollama server. Make sure it's running."
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to Ollama server at 127.0.0.1:11434. Make sure it's running with 'ollama serve'."
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error calling Ollama: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg

# ----------- Fireworks AI API Call Logic -----------
def call_fireworks_api(prompt, context, pdf_path=None):
    API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
    }
    
    message_content = [
        {
            "type": "text",
            "text": f"Context from PDF: {context}\n\nUser question: {prompt}\n\nAnswer the question based on the provided context."
        }
    ]
    
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

# ----------- Central Response Generator with Model Fallback Logic -----------
def response_generator(text, prompt, pdf_path=None):
    # Find relevant context and check for missing information
    context, missing_info = fuzzy_match_query(text, prompt)
    
    # Ask for more info if needed
    if missing_info and len(missing_info) > 0:
        return {
            "answer": f"I'd like to help, but I need more information about the {', '.join(missing_info)} to give you a proper answer. Could you please provide more details?",
            "needs_info": True
        }
    
    # Use selected model
    if selected_model == "Local Ollama":
        try:
            logger.info(f"Using Ollama model: {ollama_model}")
            # Add a note about PDF-only responses
            st.info("Using local Ollama model. Responses will be based ONLY on the PDF content.")
            ollama_response = call_ollama_api(prompt, context, ollama_model, pdf_path)
            if ollama_response and not ollama_response.startswith("Error:") and not ollama_response.startswith("Could not connect"):
                return {"answer": enhance_response(ollama_response)}
            else:
                st.warning("No valid response from Ollama, falling back to other models", icon="⚠️")
                logger.warning(f"Invalid Ollama response: {ollama_response}")
        except Exception as e:
            st.warning(f"Error with Ollama: {e}. Falling back to other models.", icon="⚠️")
            logger.error(f"Exception in Ollama call: {str(e)}")
    
    if selected_model == "OpenRouter":
        try:
            logger.info("Using OpenRouter API with deepseek model")
            # Add a note about PDF-only responses
            st.info("Using OpenRouter deepseek model. Responses will be based ONLY on the PDF content.")
            openrouter_response = call_openrouter_api(prompt, context, pdf_path)
            if openrouter_response and not openrouter_response.startswith("Error:"):
                return {"answer": enhance_response(openrouter_response)}
            else:
                st.warning("No valid response from OpenRouter, falling back to other models", icon="⚠️")
                logger.warning(f"Invalid OpenRouter response: {openrouter_response}")
        except Exception as e:
            st.warning(f"Error with OpenRouter: {e}. Falling back to other models.", icon="⚠️")
            logger.error(f"Exception in OpenRouter call: {str(e)}")
    
    if selected_model == "Fireworks AI" or (selected_model == "Local Ollama" and ('ollama_response' not in locals() or 
                                                                                  (ollama_response and (ollama_response.startswith("Error:") or 
                                                                                                     ollama_response.startswith("Could not connect"))))) or \
       (selected_model == "OpenRouter" and ('openrouter_response' not in locals() or 
                                           (openrouter_response and openrouter_response.startswith("Error:")))):
        try:
            logger.info("Trying Fireworks AI API")
            fireworks_response = call_fireworks_api(prompt, context, pdf_path)
            if fireworks_response:
                return {"answer": enhance_response(fireworks_response)}
        except Exception as e:
            st.warning(f"Falling back to BERT model: {e}", icon="⚠️")
            logger.error(f"Exception in Fireworks API call: {str(e)}")
    
    logger.info("Using BERT model as fallback")
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
    
    if 'answer' in output:
        output['answer'] = enhance_response(output['answer'])
    
    return output

# ----------- Streamlit Chat UI and Interaction Logic -----------
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
    
    pdf_path = os.path.join(upload_folder, uploaded_file.name) if uploaded_file else None
    
    with st.spinner("Processing your question..."):
        response = response_generator(extracted_text if extracted_text else "", prompt, pdf_path)
    
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
