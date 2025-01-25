import streamlit as st
import os
from config import setup_environment, initialize_session_state, TEMP_UPLOAD_DIR, SUPPORTED_FILE_TYPES
from chat_interface import display_chat_interface, handle_sidebar
from document_loader import process_document, load_db
from error_handling import handle_error, validate_file
from langchain.schema import Document

def handle_document_upload():
    """Handle document upload in the Documents tab."""
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'csv', 'txt', 'docx', 'pptx', 'xlsx'],
        accept_multiple_files=True,
        help="Upload one or more documents to start chatting"
    )
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_docs]
        if new_files and st.button(f"Process {len(new_files)} Document(s)", key="process_docs"):
            with st.spinner("Processing documents..."):
                success = True
                for file in new_files:
                    temp_path = os.path.join(TEMP_UPLOAD_DIR, file.name)
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(file.getvalue())
                        if not process_document(temp_path, file.name.split('.')[-1].lower()):
                            success = False
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                if success:
                    st.success("Documents processed successfully!")
                    st.rerun()

def handle_youtube_input():
    """Handle YouTube URL input."""
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    if youtube_url and youtube_url not in st.session_state.uploaded_docs:
        if st.button("Process Video", key="process_video"):
            with st.spinner("Processing video..."):
                if process_document(youtube_url, 'youtube'):
                    st.success("Video processed successfully!")
                    st.rerun()

def handle_website_input():
    """Handle website URL input."""
    web_url = st.text_input(
        "Website URL",
        placeholder="https://..."
    )
    if web_url and web_url not in st.session_state.uploaded_docs:
        if st.button("Process Website", key="process_web"):
            with st.spinner("Processing website..."):
                if process_document(web_url, 'url'):
                    st.success("Website processed successfully!")
                    st.rerun()

def combine_documents(doc_keys):
    """Combine selected documents into a single QA chain."""
    try:
        all_documents = []
        for key in doc_keys:
            all_documents.extend(st.session_state.uploaded_docs[key]["documents"])
        
        # Create new QA chain with combined documents
        qa = load_db(all_documents)
        if qa:
            # Create a descriptive name for the combination
            doc_names = [os.path.basename(k) if not k.startswith(("YouTube:", "Web:")) else k.split(": ")[1] for k in doc_keys]
            combined_key = "Combined: " + ", ".join(doc_names)
            
            st.session_state.uploaded_docs[combined_key] = {
                "qa": qa,
                "documents": all_documents
            }
            st.session_state.qa = qa
            st.session_state.current_doc = combined_key
            return True
        return False
    except Exception as e:
        handle_error("Error combining documents", e)
        return False

def format_document_name(doc_key):
    """Format document name for display."""
    if doc_key.startswith("YouTube: 📺"):
        return doc_key.split("YouTube: ")[1]  # Already has emoji
    elif doc_key.startswith("Web: 🌐"):
        return doc_key.split("Web: ")[1]  # Already has emoji
    elif doc_key.startswith("Combined:"):
        return "📚 " + doc_key.split("Combined: ")[1]
    else:
        # Determine file type and add appropriate icon
        file_ext = os.path.splitext(doc_key)[1].lower()
        icons = {
            '.pdf': '📄',
            '.docx': '📝',
            '.doc': '📝',
            '.txt': '📋',
            '.csv': '📊',
            '.xlsx': '📊',
            '.xls': '📊',
            '.pptx': '📎',
            '.ppt': '📎'
        }
        icon = icons.get(file_ext, '📄')
        return f"{icon} {os.path.basename(doc_key)}"

def main():
    """Main application function."""
    st.set_page_config(
        page_title="FileWhisperer",
        page_icon="🦉",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    setup_environment()
    initialize_session_state()
    handle_sidebar()

    # Document selector in sidebar if documents are loaded
    if st.session_state.uploaded_docs:
        st.sidebar.markdown("### 📚 Available Documents")
        
        # Create two columns for the chat mode selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            single_mode = st.radio("Chat Mode", ["Single", "Multiple"])
        
        available_docs = list(st.session_state.uploaded_docs.keys())
        
        if single_mode == "Single":
            # Single document selection
            selected_doc = st.sidebar.selectbox(
                "Select a document to chat with",
                options=available_docs,
                format_func=format_document_name,
                index=available_docs.index(st.session_state.current_doc) if st.session_state.current_doc in available_docs else 0
            )
            
            if selected_doc != st.session_state.current_doc:
                st.session_state.qa = st.session_state.uploaded_docs[selected_doc]["qa"]
                st.session_state.current_doc = selected_doc
                st.rerun()
        
        else:
            # Multiple document selection
            selected_docs = st.sidebar.multiselect(
                "Select documents to chat with",
                options=available_docs,
                default=[st.session_state.current_doc] if st.session_state.current_doc else None,
                format_func=format_document_name
            )
            
            if len(selected_docs) > 1:
                if st.sidebar.button("💬 Chat with Selected Documents"):
                    with st.spinner("Combining documents..."):
                        if combine_documents(selected_docs):
                            st.success("Documents combined successfully!")
                            st.rerun()
            elif len(selected_docs) == 1 and selected_docs[0] != st.session_state.current_doc:
                st.session_state.qa = st.session_state.uploaded_docs[selected_docs[0]]["qa"]
                st.session_state.current_doc = selected_docs[0]
                st.rerun()

    # Main content area
    if st.session_state.qa:
        current_doc = st.session_state.current_doc
        if current_doc.startswith("Combined:"):
            st.markdown(f"### 🗣️ Chatting with Multiple Documents")
            st.markdown(f"*Documents: {current_doc.replace('Combined: ', '')}*")
        else:
            display_name = format_document_name(current_doc)
            st.markdown(f"### 🗣️ Chatting with: {display_name}")
        
        st.markdown("*I'll use the document content when available and my general knowledge when needed!*")
        display_chat_interface()
        
        if st.button("Upload More Documents"):
            st.session_state.qa = None
            st.rerun()
    else:
        st.title("Welcome to FileWhisperer 🦉")
        
        # Create tabs for different input types
        tab1, tab2, tab3 = st.tabs(["📄 Documents", "🎥 YouTube", "🌐 Website"])
        
        with tab1:
            handle_document_upload()
        
        with tab2:
            handle_youtube_input()
        
        with tab3:
            handle_website_input()

if __name__ == "__main__":
    main()
