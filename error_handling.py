import streamlit as st
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handle_error(message: str, error: Exception, show_user: bool = True):
    """
    Centralized error handling.
    
    Args:
        message (str): Context message about where the error occurred
        error (Exception): The actual error object
        show_user (bool): Whether to display the error to the user via Streamlit
    """
    error_msg = f"{message}: {str(error)}"
    logger.error(error_msg)
    if show_user:
        st.error(error_msg)
    return None

def validate_file(file_path: str, file_type: str) -> Optional[str]:
    """
    Validate file before processing.
    
    Args:
        file_path (str): Path to the file
        file_type (str): Type of the file
        
    Returns:
        Optional[str]: Error message if validation fails, None if successful
    """
    try:
        if not file_path:
            return "File path is empty"
        if file_type not in ['pdf', 'csv', 'txt', 'docx', 'pptx', 'xlsx', 'youtube', 'url']:
            return f"Unsupported file type: {file_type}"
        return None
    except Exception as e:
        return str(e)