import streamlit as st
import base64
from error_handling import handle_error

LOGO_PATH = r"FileWhisperer_logo.png"

def get_image_base64(image_path=None):
    """Convert image to base64 string."""
    try:
        with open(LOGO_PATH, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        handle_error(f"Error loading logo from {LOGO_PATH}", e)
        return ""

def display_chat_interface():
    """Display chat interface."""
    try:
        # Display existing messages
        for message in st.session_state.chat_history.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input
        if prompt := st.chat_input("Ask anything about the documents!"):
            # Add user message
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.chat_history.add_message("user", prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.qa({
                        "question": prompt,
                        "chat_history": st.session_state.chat_history.get_langchain_format()
                    })
                    
                    response = result["answer"]
                    st.markdown(response)
                    
                    # Show sources if response was from documents
                    if result.get("source_documents") and response.startswith("🦉 Based on"):
                        with st.expander("View Sources"):
                            for doc in result["source_documents"]:
                                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                st.markdown(f"```\n{doc.page_content}\n```")
                    
                    st.session_state.chat_history.add_message("assistant", response)
    except Exception as e:
        handle_error("Error in chat interface", e)

def handle_sidebar():
    """Handle sidebar chat history management with improved styling."""
    with st.sidebar:
        # Get base64 encoded logo
        logo_base64 = get_image_base64()
        
        st.markdown(
            f"""
            <div style='text-align: center; padding: 1rem 0;'>
                <img src='data:image/png;base64,{logo_base64}' width='180px' style='margin-bottom: 1rem; border-radius: 10px;'>
                <h1 style='font-size: 2rem; margin: 0.5rem 0; color: #2E4057;'>FileWhisperer</h1>
                <p style='font-size: 1.1rem; color: #666; margin-bottom: 1.5rem;'>Your AI Document Assistant</p>
                <hr style='margin: 1rem 0; border: none; height: 1px; background: linear-gradient(to right, transparent, #2E4057, transparent);'>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown("### Chat History")
        if st.button("Clear History"):
            st.session_state.chat_history.clear_history()
            st.rerun()
