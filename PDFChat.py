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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PDFChat')

# ----------- Upload Folder Setup -----------
upload_folder = 'uploaded_pdf_file'

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

# ----------- Streamlit UI: Header and Sidebar -----------
st.header("PDF Chatbot")

# Sidebar for model selection and usage guide
st.sidebar.title("Model Settings")
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Local Ollama"

selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    ["Local Ollama"],
    index=0,
    key="model_selector"
)
st.session_state.selected_model = selected_model

# Add usage guide in sidebar
st.sidebar.markdown("---")
st.sidebar.title("How to Use This System")

st.sidebar.markdown("""
### 📚 General Usage
1. **Ask Questions**: Type your question in the chat box
2. **Upload PDFs**: Use the file uploader to add PDF documents
3. **Choose Model**: Select your preferred AI model from the dropdown

### 🚗 Car Recommendations
- For students: Budget limit of RM 50,000
- All recommendations include specific prices
- Recommendations are based on PDF content

### ⚙️ Model Options
- **Local Ollama**: 
  - llama3.1:8b: More powerful but slower
  - qwen3:8b: Fast processing. Useful for high-level logical reasoning
  - gemma3:4b: Competitive performance in mathematics.
  - phi3:3.8b: Clear and concise explanations.

### 💡 Tips
- Upload PDFs for specific document-based answers
- Ask general questions without uploading files
- Be specific in your questions for better answers
- Check the model status in the sidebar
""")

# ----------- Ollama Connection Check -----------
if st.session_state.selected_model == "Local Ollama":
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

    # Ollama model selection with available models
    if 'ollama_model' not in st.session_state:
        st.session_state.ollama_model = "llama3.1:8b"
    
    ollama_model = st.sidebar.selectbox(
        "Choose Ollama Model",
        ["llama3.1:8b", "deepseek-r1:1.5b","qwen3:8b","gemma3:4b","phi3:3.8b"],
        index=0,
        key="ollama_model_selector"
    )
    st.session_state.ollama_model = ollama_model
    
    # Add model information
    if st.session_state.ollama_model == "llama3.1:8b":
        st.sidebar.info("Using llama3.1:8b - Powerful model with excellent reasoning capabilities")
        st.sidebar.warning("This is a large model and may take longer to process. Please be patient.", icon="⚠️")
    elif st.session_state.ollama_model == "deepseek-r1:1.5b":
        st.sidebar.info("Using deepseek-r1:1.5b - Fast and efficient model for quick responses")
    elif st.session_state.ollama_model == "qwen3:8b":
        st.sidebar.info("Using qwen3:8b - Powerful model with excellent reasoning capabilities")
    elif st.session_state.ollama_model == "gemma3:4b":
        st.sidebar.info("Using gemma3:4b - Competitive performance in mathematics.")
    elif st.session_state.ollama_model == "phi3:3.8b":
        st.sidebar.info("Using phi3:3.8b - Clear and concise explanations.")
        
    
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
                        if st.session_state.ollama_model in model_names:
                            st.sidebar.success(f"✅ Model '{st.session_state.ollama_model}' is available!")
                        else:
                            st.sidebar.warning(f"⚠️ Model '{st.session_state.ollama_model}' not found. Available models: {', '.join(model_names)}")
                            st.sidebar.info(f"Try: ollama pull {st.session_state.ollama_model}")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not check model availability: {str(e)}")
            else:
                st.sidebar.error("❌ Ollama server responded with an error")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"❌ Could not connect to Ollama server: {str(e)}")
            st.sidebar.info("Make sure Ollama is running with 'ollama serve' command")

# ----------- PDF Upload and Text Extraction -----------
def clean_text(text):
    """Clean and format extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'-\s*\n\s*', '', text)  # Fix hyphenated words
    text = re.sub(r'\n\s*(?=[a-z])', ' ', text)  # Fix line breaks in sentences
    text = re.sub(r'\n\s*(?=[A-Z])', '. ', text)  # Fix line breaks between sentences
    
    # Remove headers and footers (common in PDFs)
    text = re.sub(r'\n\d+\s*\n', '\n', text)  # Remove page numbers
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)  # Remove page numbers
    text = re.sub(r'©.*?\n', '', text)  # Remove copyright notices
    text = re.sub(r'Confidential.*?\n', '', text)  # Remove confidentiality notices
    
    # Clean up remaining whitespace
    text = text.strip()
    
    return text

def extract_text_from_pdf(pdf_path):
    """Enhanced PDF text extraction with better formatting and error handling"""
    try:
        reader = PdfReader(pdf_path)
        number_of_pages = len(reader.pages)
        
        # Extract text from all pages with metadata
        pdf_text = ""
        page_metadata = []
        
        for i in range(number_of_pages):
            page = reader.pages[i]
            
            # Extract text
            page_text = page.extract_text()
            
            # Clean the extracted text
            cleaned_text = clean_text(page_text)
            
            # Extract metadata
            metadata = {
                'page_number': i + 1,
                'page_size': page.mediabox,
                'has_images': len(page.images) > 0,
                'text_length': len(cleaned_text)
            }

            # Extract car data using regular expressions
            car_pattern = re.compile(
                r"(●\s*.*?\n"  # Car model, handles potential leading bullet
                r"○\s*Price:\s*(.*?)\n"
                r"○\s*Basic Description:\s*(.*?)\n"
                r"○\s*Engine Type and Size:\s*(.*?)\n"
                r"(?:○\s*Power Output:\s*(.*?)\n)?"  # Optional power output
                r"(?:○\s*Torque Output:\s*(.*?)\n)?"
                r"(?:○\s*Transmission Type:\s*(.*?)\n)?"
                r"Key Features:\s*(.*?)(?=\n●|\Z))",
                re.DOTALL | re.MULTILINE
            )
            car_matches = car_pattern.findall(page_text)
                
            cars = []
            for match in car_matches:
                if len(match) == 8:
                    (model, price, description, engine, power, torque, transmission, features) = match
                elif len(match) == 4:
                    (model, price, description, engine) = match
                    power, torque, transmission, features = None, None, None, None
                elif len(match) == 2:
                    model = match[1].strip()
                    price, description, engine, power, torque, transmission, features = None, None, None, None, None, None, None
                else:
                    logging.warning(f"Skipping malformed car data match on page {i + 1}: {match}")
                    continue
                    
                cars.append({
                    'model': model.replace("●","").strip(),
                    'price': price.strip(),
                    'description': description.strip(),
                    'engine': engine.strip(),
                    'power': power.strip() if power else None,
                    'torque': torque.strip() if torque else None,
                    'transmission': transmission.strip() if transmission else None,
                    'features': features.strip().replace('\n', ', ') if features else None,
                })
            metadata['cars'] = cars # Store car data in page metadata
            
            # Add page separator and metadata
            pdf_text += f"\n\n--- Page {i + 1} ---\n\n{cleaned_text}"
            page_metadata.append(metadata)
        
        return pdf_text, page_metadata
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return "", []

uploaded_files = st.file_uploader("Choose PDF files", type=['pdf','PDF'], accept_multiple_files=True)

extracted_text = ""
if uploaded_files:
    # Create a list to store all PDF texts and metadata
    all_pdf_texts = []
    all_metadata = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        saved_path = os.path.join(upload_folder, file_name)

        with open(saved_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"PDF file '{file_name}' has successfully uploaded to {saved_path}")

        # Extract text and metadata
        pdf_text, metadata = extract_text_from_pdf(saved_path)
        
        if pdf_text:
            all_pdf_texts.append(pdf_text)
            all_metadata.append({
                'filename': file_name,
                'pages': metadata,
                'total_pages': len(metadata),
                'has_images': any(page['has_images'] for page in metadata)
            })
            
            # Display document summary
            with st.expander(f"Document Summary: {file_name}"):
                st.write(f"Total Pages: {len(metadata)}")
                st.write(f"Contains Images: {'Yes' if any(page['has_images'] for page in metadata) else 'No'}")
                st.write(f"Average Text per Page: {sum(page['text_length'] for page in metadata) // len(metadata)} characters")
                
                # Show first page preview
                st.write("First Page Preview:")
                st.write(pdf_text.split("--- Page 1 ---")[1].split("--- Page 2 ---")[0] if "--- Page 2 ---" in pdf_text else pdf_text)

        else:
            st.warning(f"Could not extract text from {file_name}")
    
    # Combine all PDF texts with clear separation
    if all_pdf_texts:
        extracted_text = "\n\n--- Document Separator ---\n\n".join(all_pdf_texts)
        
        # Show summary of uploaded documents
        st.info(f"Successfully processed {len(uploaded_files)} PDF documents with a total of {sum(meta['total_pages'] for meta in all_metadata)} pages.")
        
        # Store metadata in session state for later use
        st.session_state.pdf_metadata = all_metadata
    else:
        st.error("No text could be extracted from any of the uploaded PDFs.")

# ----------- Response Enhancement Function -----------
def enhance_response(answer, source_doc=None, page_number=None):
    """Enhanced response formatting with source attribution"""
    # Add source information if available
    source_info = ""
    if source_doc:
        source_info = f"Based on document '{source_doc}'"
        if page_number:
            source_info += f", page {page_number}"
        source_info += ":\n\n"
    
    # Format the response
    formatted_response = f"{source_info}{answer}"
    
    return formatted_response

# ----------- Fuzzy Matching Logic for Context Extraction -----------
def fuzzy_match_query(text, query):
    key_terms = re.findall(r'\b\w+\b', query.lower())
    key_terms = [term for term in key_terms if len(term) > 3]
    
    # Split text into documents
    documents = text.split("\n\n--- Document Separator ---\n\n")
    best_paragraph = ""
    highest_score = 0
    source_doc = None
    page_number = None
    
    for doc_idx, doc_text in enumerate(documents):
        # Split document into pages
        pages = re.split(r'--- Page \d+ ---', doc_text)
        for page_idx, page_text in enumerate(pages[1:], 1):  # Skip first empty split
            paragraphs = page_text.split('\n\n')
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
                    source_doc = uploaded_files[doc_idx].name if uploaded_files else None
                    page_number = page_idx
    
    # Check if we found any relevant content
    if highest_score == 0:
        return "", ["content not found in PDF"], None, None
    
    context = best_paragraph
    missing_info = []
    if "when" in query.lower() and not re.search(r'\b(date|day|month|year|time)\b', query, re.IGNORECASE):
        missing_info.append("time period")
    if "where" in query.lower() and not re.search(r'\b(location|place|address|city)\b', query, re.IGNORECASE):
        missing_info.append("location")
        
    return context, missing_info, source_doc, page_number

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

def generate_chat_history_pdf(chat_history):
    """
    Generates a PDF from the chat history.

    Args:
        chat_history (list): A list of dictionaries, where each dictionary
            represents a message and has keys "role" and "content".

    Returns:
        bytes: The PDF content as bytes.
    """
    buffer = BytesIO()  # Use BytesIO for in-memory PDF generation
    p = canvas.Canvas(buffer, pagesize=letter)
    textobject = p.beginText()
    textobject.setFont("Helvetica", 12)
    y = 750  # Starting Y position

    for message in chat_history:
        role = message["role"].capitalize()
        content = message["content"]
        textobject.textLine(f"{role}: {content}")
        y -= 15  # Adjust spacing

        if y < 50:  # Start a new page if we're near the bottom
            p.drawText(textobject)
            p.showPage()
            textobject = p.beginText()
            textobject.setFont("Helvetica", 12)
            y = 750

    p.drawText(textobject)
    p.save()
    buffer.seek(0)  # Go back to the beginning of the buffer
    return buffer.getvalue()

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
def call_ollama_api(prompt, context, model="llama3.1:8b", pdf_path=None):
    """Call the Ollama API with the specified model"""
    API_URL = "http://127.0.0.1:11434/api/chat"
    
    logger.info(f"Calling Ollama API with model: {model}")
    
    # Prepare messages with balanced instructions
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that primarily uses the provided content to answer questions, while also providing relevant additional information when helpful.

You have access to the following context: {context}

Your task is to:
1. First, analyze the entire content carefully
2. Base your answer primarily on the information found in the content
3. If the content is limited, just give answer based on info given.
4. Structure your response in a clear and organized manner
5. Use bullet points or numbered lists for better readability
6. For car recommendations, always include specific prices if available.
7. Only recommend models that are in the content, no need to add on any other models.
8. For the price in PDF, usually it is in format of RM XXX,XXX.00 , so please get the entire price, not just the last 3 digits.
9. Cannot answer any question that are not related to cars.
10.  Answer questions *only* about cars, car models, car features, and car specifications found in the content.
11.  Do not answer questions that are not related to cars.
12.  If the answer is not in the content, respond with "I cannot answer questions not related to cars."

Remember: The content is your main source."""
        },
        {
            "role": "user",
            "content": f"Please provide an answer based primarily on the content, supplemented with relevant additional information if needed. {prompt}"
        }
    ]
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.8,
            "num_predict": 2048
        }
    }
    
    try:
        logger.info(f"Sending request to Ollama API at {API_URL}")
        # Adjust timeout based on model size
        timeout = 18000 if model == "llama3.1:8b" else 6000
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

# ----------- Central Response Generator with Model Fallback Logic -----------
def response_generator(text, prompt, pdf_path=None):
    # Reject questions if no PDF content is provided
    if not text or text.strip() == "":
        return {
            "answer": "I'm sorry, I can only answer questions that are based on the uploaded PDF content. Please upload a PDF first.",
            "needs_info": True
        }

    # Extract relevant context from the PDF
    context, missing_info, source_doc, page_number = fuzzy_match_query(text, prompt)

    # Reject if the question isn't related to the content
    if not context or "content not found in PDF" in missing_info:
        return {
            "answer": "I'm sorry, I couldn't find any relevant information in the uploaded PDF to answer your question.",
            "needs_info": True
        }

    # Reject if important keywords are missing
    if missing_info and len(missing_info) > 0:
        return {
            "answer": f"I need more details to find a matching answer in the PDF. Missing info: {', '.join(missing_info)}.",
            "needs_info": True
        }

    # Only respond using PDF-based context
    try:
        if st.session_state.selected_model == "Local Ollama":
            logger.info(f"Using Ollama model: {st.session_state.ollama_model}")
            st.info("Using local Ollama model. Only PDF-based questions are allowed.")

            if st.session_state.ollama_model == "llama3.1:8b":
                st.warning("Using the larger llama3.1:8b model. This may take longer to process.", icon="⚠️")

            ollama_response = call_ollama_api(prompt, context, st.session_state.ollama_model, pdf_path)
            if ollama_response and not ollama_response.startswith("Error:") and not ollama_response.startswith("Could not connect"):
                return {"answer": enhance_response(ollama_response)}
            else:
                st.warning("No valid response from Ollama", icon="⚠️")
                logger.warning(f"Invalid Ollama response: {ollama_response}")
    except Exception as e:
        logger.error(f"Ollama processing error: {str(e)}")
        return {
            "answer": "Sorry, I encountered an issue while processing the PDF content. Please try again.",
            "needs_info": True
        }

    return {
        "answer": "I couldn't process your request. Please ensure it's directly related to the uploaded PDF content.",
        "needs_info": True
    }


# ----------- Streamlit Chat UI and Interaction Logic -----------
st.title("PDF Chat Assistant")

# Add initial greeting only if not already added
if "messages" not in st.session_state:
    st.session_state.messages = []

# Only add greeting if the chat is completely empty
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm your PDF Chat Assistant. How can I help you today? 😊"
    })


# Render chat history (ONLY ONCE)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Response logic
    lower_prompt = prompt.lower().strip()

    if lower_prompt in ["hello", "hi"]:
        response_text = "Hi there! How can I assist you today?"
    elif "how are you" in lower_prompt:
        response_text = "I'm just a program, but I'm here to help you!"
    elif "help" in lower_prompt:
        response_text = "You can ask me anything about the PDFs you upload."
    elif "bye" in lower_prompt:
        response_text = "Goodbye! Have a great day! 😊"
    else:
        pdf_path = None
        if uploaded_files and len(uploaded_files) > 0:
            pdf_path = os.path.join(upload_folder, uploaded_files[0].name)

        with st.spinner("Processing your question..."):
            response = response_generator(extracted_text if extracted_text else "", prompt, pdf_path)

        response_text = response['answer']

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

# View extracted text
if extracted_text:
    with st.expander("View Extracted Text"):
        st.text_area("Extracted Text", extracted_text, height=300)
