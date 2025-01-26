from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader,
    TextLoader, WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
from config import MODEL_NAME, TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP, get_llm
from error_handling import handle_error
import os
import tempfile
from openai import OpenAI

class DocumentLoaderFactory:
    """Factory for creating document loaders."""
    
    @staticmethod
    def get_loader(file_path: str, file_type: str):
        loaders = {
            'pdf': PyPDFLoader,
            'csv': CSVLoader,
            'pptx': UnstructuredPowerPointLoader,
            'docx': UnstructuredWordDocumentLoader,
            'xlsx': UnstructuredExcelLoader,
            'txt': TextLoader
        }
        return loaders[file_type](file_path)

def load_db(documents: list, chain_type: str = "stuff", k: int = 4):
    """Create QA chain from documents."""
    try:
        # Get LLM instance from config
        llm = get_llm()
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        prompt_template = """You are a helpful AI assistant with a 🦉 logo. Answer questions based on the provided context. If the context doesn't contain the answer, use your general knowledge but clearly indicate this.

        Context: {context}
        Question: {question}

        Instructions:
        1. If the answer is found in the context:
           - Use the context to provide a detailed answer
           - Start your response with "🦉 Based on the documents:"
        2. If the answer is NOT found in the context:
           - Use your general knowledge to answer
           - Start your response with "🦉 Based on my knowledge:"
        3. Always aim to be helpful and accurate
        4. If you're unsure, be honest about it

        Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

        # Create the chain using the LLM from config
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type=chain_type,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        return qa
    except Exception as e:
        handle_error("Error creating QA chain", e)
        return None

def transcribe_video(video_path):
    """Transcribe video using OpenAI Whisper API."""
    try:
        client = OpenAI()  # This will use your OPENAI_API_KEY from environment
        
        with open(video_path, "rb") as video_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=video_file,
                response_format="text"
            )
        return transcript
    except Exception as e:
        handle_error("Error transcribing video", e)
        return None

def process_document(file_path: str, file_type: str):
    """Process document with chunking."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Load and split document
        if file_type == 'csv':
            df = pd.read_csv(file_path)
            text = df.to_string()
            texts = text_splitter.split_text(text)
            documents = [Document(page_content=t, metadata={"source": file_path}) for t in texts]
        elif file_type == 'youtube':
            video_id = file_path.split('v=')[-1]
            # Get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Try to get video title, fallback to video ID if not available
            try:
                response = requests.get(f"https://www.youtube.com/watch?v={video_id}")
                soup = BeautifulSoup(response.text, 'html.parser')
                video_title = soup.find('title').text.replace(' - YouTube', '')
            except:
                video_title = f"YouTube Video ({video_id})"
            
            # Process transcript
            text = ' '.join([entry['text'] for entry in transcript])
            texts = text_splitter.split_text(text)
            
            source_key = f"YouTube: 📺 {video_title}"
            documents = [Document(page_content=t, metadata={"source": source_key}) for t in texts]
        elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
            # Get video name
            video_name = os.path.basename(file_path)
            
            # Transcribe video
            transcript = transcribe_video(file_path)
            if not transcript:
                st.error("Failed to transcribe video")
                return False
            
            # Split transcript into chunks
            texts = text_splitter.split_text(transcript)
            source_key = f"Video: 🎥 {video_name}"
            documents = [Document(page_content=t, metadata={"source": source_key}) for t in texts]
        elif file_type == 'url':
            # Initialize WebBaseLoader with headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Get webpage title first
            try:
                response = requests.get(file_path, headers=headers, verify=False)
                soup = BeautifulSoup(response.text, 'html.parser')
                page_title = soup.title.string.strip() if soup.title else "Webpage"
            except Exception as e:
                st.warning(f"Could not fetch page title: {str(e)}")
                page_title = "Webpage"
            
            # Load content
            loader = WebBaseLoader(
                web_paths=[file_path],
                header_template=headers,
                verify_ssl=False,
                requests_per_second=2
            )
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            
            source_key = f"Web: 🌐 {page_title}"
            for doc in texts:
                doc.metadata["source"] = source_key
            documents = texts
        else:
            loader = DocumentLoaderFactory.get_loader(file_path, file_type)
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            for doc in texts:
                doc.metadata["source"] = file_path
            documents = texts

        # Create QA chain
        qa = load_db(documents)
        if qa:
            # Use the source_key for YouTube and Web sources, otherwise use file_path
            doc_key = source_key if file_type in ['youtube', 'url', 'mp4', 'avi', 'mov', 'mkv'] else file_path
            st.session_state.uploaded_docs[doc_key] = {
                "qa": qa,
                "documents": documents
            }
            st.session_state.qa = qa
            st.session_state.current_doc = doc_key
            return True
        return False
        
    except Exception as e:
        handle_error("Error processing document", e)
        return False
