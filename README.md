# Chat with Your Document

This Streamlit application allows you to chat with the content of a PDF or text document using a Language Model (LLM).

## How to Use

1.  Upload a PDF or TXT file.
2.  Type your question in the chat input and press Enter.
3.  The LLM will provide a response based on the document content.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/1165qwc/Chatbot3.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd Chatbot
    ```

3.  Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

4.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Set your OpenAI API key. for streamlit sharing, create a secrets.toml file, otherwise set it in environment variables.

6.  Run the Streamlit app:

    ```bash
    streamlit run Chatbot3.py
    ```

## Notes

* Replace `<your-repository-url>` with the actual URL of your GitHub repository.
* Remember to set your OpenAI API key as an environment variable or using Streamlit secrets.
* For streamlit sharing, create a `secrets.toml` file in the root directory.
