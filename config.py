import os
from dotenv import load_dotenv
import nltk
from chat_history import ChatHistory
import streamlit as st
from langchain_openai import ChatOpenAI
import logging

# Suppress warnings
logging.getLogger('langchain_community.utils.user_agent').setLevel(logging.ERROR)

# Load environment variables with override
load_dotenv(override=True)

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data silently."""
    try:
        # Set NLTK data path
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

        # List of required NLTK resources
        resources = [
            'punkt',
            'punkt_tab',  # Added this specific resource
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words',
            'stopwords',
            'wordnet',
            'omw-1.4'
        ]
        
        # Download each resource silently
        for resource in resources:
            try:
                if not nltk.data.find(resource, paths=[nltk_data_dir]):
                    nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            except Exception:
                try:
                    # Fallback: try downloading without specific path
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    print(f"Failed to download {resource}: {str(e)}")
                
    except Exception as e:
        print(f"Error setting up NLTK: {str(e)}")

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
    # Set environment variable to suppress user agent warning
    os.environ['LANGCHAIN_USER_AGENT'] = 'FileWhisperer/1.0'
    
    # Download NLTK data silently
    download_nltk_data()
    
    # Create necessary directories
    for directory in [CHAT_HISTORY_DIR, TEMP_UPLOAD_DIR, TEMP_URL_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Verify OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found in environment variables.")
        st.stop()

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
