import os
from dotenv import load_dotenv
import nltk
from chat_history import ChatHistory
import streamlit as st
from langchain_openai import ChatOpenAI

# Load environment variables with override
load_dotenv(override=True)

# Constants
CHAT_HISTORY_DIR = "chat_histories"
TEMP_UPLOAD_DIR = "temp_uploads"
TEMP_URL_DIR = "temp_urls"
MODEL_NAME = "gpt-4"
TEMPERATURE = 0
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Supported file types
SUPPORTED_FILE_TYPES = {
    'pdf': 'PDF Documents',
    'csv': 'CSV Files',
    'txt': 'Text Files',
    'docx': 'Word Documents',
    'pptx': 'PowerPoint Presentations',
    'xlsx': 'Excel Spreadsheets',
    'youtube': 'YouTube Videos',
    'url': 'Websites'
}

def get_llm():
    """Create a new LLM instance each time"""
    return ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

def setup_environment():
    """Setup environment and dependencies."""
    # Verify OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found in environment variables.")
        st.stop()
    
    # Create necessary directories
    for directory in [CHAT_HISTORY_DIR, TEMP_UPLOAD_DIR, TEMP_URL_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    # Create temp directory for uploads
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    return TEMP_UPLOAD_DIR

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatHistory()
    if 'qa' not in st.session_state:
        st.session_state.qa = None
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = {}
    if 'current_doc' not in st.session_state:
        st.session_state.current_doc = None