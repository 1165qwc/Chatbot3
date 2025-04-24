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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PDFChat')

# Check if API keys are available
if not OPENROUTER_API_KEY:
    logger.warning("OpenRouter API key not found in environment variables")
    OPENROUTER_API_KEY = "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"  # Fallback to hardcoded key

if not FIREWORKS_API_KEY:
    logger.warning("Fireworks API key not found in environment variables")
    FIREWORKS_API_KEY = "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"  # Fallback to hardcoded key

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
        ["deepseek-r1:1.5b", "deepseek-r1:8b"],
        index=0
    )
    
    # Add model information
    if ollama_model == "deepseek-r1:8b":
        st.sidebar.info("Using deepseek-r1:8b - This is a larger model with better reasoning capabilities")
    else:
        st.sidebar.info("Using deepseek-r1:1.5b - Faster but less capable than the 8b model")
    
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
    
    # Check if API key is available
    if not OPENROUTER_API_KEY:
        error_msg = "OpenRouter API key is missing. Please set the OPENROUTER_API_KEY environment variable."
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg
    
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
                "HTTP-Referer": "https://github.com/1165qwc/Chatbot3/edit/main/PDFChat.py",  # Add a referer header
                "X-Title": "PDF Chatbot"  # Add a title header
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
                st.error(f"OpenRouter API error: {response.text}")
            else:
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
    
    # Check if we found any relevant content
    if highest_score == 0:
        return "", ["content not found in PDF"]
    
    context = best_paragraph
    missing_info = []
    if "when" in query.lower() and not re.search(r'\b(date|day|month|year|time)\b', query, re.IGNORECASE):
        missing_info.append("time period")
    if "where" in query.lower() and not re.search(r'\b(location|place|address|city)\b', query, re.IGNORECASE):
        missing_info.append("location")
        
    return context, missing_info

# ----------- Content Validation Function -----------
def validate_response(response, pdf_content):
    """Check if the response contains information that's likely not from the PDF"""
    # Common phrases that indicate external knowledge
    external_indicators = [
        "generally speaking", "in general", "typically", "usually", 
        "as a rule", "in most cases", "according to experts", 
        "research shows", "studies indicate", "it is known that",
        "it is common knowledge", "it is widely accepted", "it is a fact that"
    ]
    
    # Check for external knowledge indicators
    for indicator in external_indicators:
        if indicator.lower() in response.lower():
            return False, f"The response contains phrases like '{indicator}' which suggest it's using external knowledge rather than PDF content only."
    
    # Check if response length is too short to be meaningful
    if len(response.strip()) < 20:
        return False, "The response is too short to be meaningful."
    
    # Check if response contains "I cannot find" which is our indicator for content not in PDF
    if "cannot find" in response.lower() or "not found in the pdf" in response.lower():
        return True, "The response correctly indicates that information is not in the PDF."
    
    # Check for key terms from the PDF content
    pdf_terms = re.findall(r'\b\w{5,}\b', pdf_content.lower())
    pdf_terms = [term for term in pdf_terms if term not in ["about", "which", "their", "there", "these", "those", "would", "could", "should"]]
    
    # Count how many PDF terms appear in the response
    pdf_term_count = 0
    for term in pdf_terms[:20]:  # Limit to first 20 terms to avoid excessive checking
        if term in response.lower():
            pdf_term_count += 1
    
    # If response has very few PDF terms, it might be using external knowledge
    if pdf_term_count < 2 and len(response.split()) > 20:
        return False, "The response contains very few terms from the PDF content, suggesting it might be using external knowledge."
    
    return True, "The response appears to be based on PDF content."

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
    
    # Adjust timeout based on model size
    timeout = 180 if model == "deepseek-r1:8b" else 60
    logger.info(f"Using timeout of {timeout} seconds for model {model}")
    
    # Enhanced system prompt with stricter instructions
    system_prompt = f"""You are a PDF assistant that ONLY answers questions based on the content of the uploaded PDF document. 
DO NOT use any external knowledge. If the answer cannot be found in the PDF, say 'I cannot find that information in the PDF.'
You have access to the following context from the PDF: {context}

Your task is to:
1. Analyze the context carefully
2. Answer the user's question using ONLY information from the PDF
3. If the answer is not in the PDF, clearly state that
4. Provide specific references to the PDF content when possible
5. NEVER make up information or use knowledge outside the PDF
6. If you're unsure, say "I cannot find that information in the PDF" rather than guessing

Remember: Your only source of information is the PDF content provided above. Do not use any pre-trained knowledge."""
    
    messages = [
        {"role": "system", "content": system_prompt},
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
        # Adjust parameters based on model size
        if model == "deepseek-r1:8b":
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1024
                }
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
    
    try:
        logger.info(f"Sending request to Ollama API at {API_URL} with timeout {timeout}")
        # Explicitly set both connect and read timeouts
        response = requests.post(API_URL, json=payload, timeout=(timeout, timeout))
        logger.info(f"Ollama API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info("Received successful response from Ollama")
            
            # Handle different API response formats
            if "message" in data:
                response_text = data["message"]["content"]
            elif "response" in data:
                response_text = data["response"]
            else:
                logger.warning("Unexpected response format from Ollama")
                return "No response from Ollama"
            
            # Validate the response to ensure it's based on PDF content
            is_valid, validation_message = validate_response(response_text, context)
            if not is_valid:
                logger.warning(f"Response validation failed: {validation_message}")
                # Return a corrected response
                return f"I cannot find that information in the PDF. {validation_message}"
            
            return response_text
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
    except requests.exceptions.ReadTimeout:
        error_msg = f"Request timed out after {timeout} seconds. The {model} model may be too large for your system or taking too long to process. Try using a smaller model like deepseek-r1:1.5b."
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
# def call_fireworks_api(prompt, context, pdf_path=None):
#     API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
#     headers = {
#               "Authorization": f"Bearer {FIREWORKS_API_KEY}",
#     }
    
#     message_content = [
#         {
#             "type": "text",
#             "text": f"Context from PDF: {context}\n\nUser question: {prompt}\n\nAnswer the question based on the provided context."
#         }
#     ]
    
#     if pdf_path:
#         image_url = get_pdf_image(pdf_path)
#         if image_url:
#             message_content.append(
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": image_url
#                     }
#                 }
#             )
    
#     payload = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": message_content
#             }
#         ],
#         "max_tokens": 512,
#         "model": "accounts/fireworks/models/llama4-scout-instruct-basic"
#     }
    
#     response = requests.post(API_URL, headers=headers, json=payload)
#     if response.status_code == 200:
#         result = response.json()
#         return result["choices"][0]["message"]["content"]
#     else:
#         raise Exception(f"API error: {response.status_code}")

# ----------- Central Response Generator with Model Fallback Logic -----------
def response_generator(text, prompt, pdf_path=None):
    # Find relevant context and check for missing information
    context, missing_info = fuzzy_match_query(text, prompt)
    
    # Check if the question is not within the PDF content
    if "content not found in PDF" in missing_info:
        return {
            "answer": f"I cannot find any information related to your question in the PDF document. The question appears to be outside the scope of the document's content. Please try asking a different question that relates to the information in the PDF.",
            "needs_info": True
        }
    
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
            
            # Add a warning for the 8b model
            if ollama_model == "deepseek-r1:8b":
                st.warning("Using the larger 8b model. This may take longer to process. If it times out, try switching to the 1.5b model.", icon="⚠️")
            
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
