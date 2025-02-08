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
from langchain.schema import Document, BaseRetriever
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
from urllib.parse import urlparse, parse_qs
import time
try:
    from pytube import YouTube
except ImportError as e:
    st.error("Pytube is not installed or there is an import error.")
    handle_error("Import error for Pytube", e)
#from pytube import YouTube
from concurrent.futures import ThreadPoolExecutor
from thefuzz import fuzz
from thefuzz import process
import numpy as np
from typing import List, Any
from pydantic import Field, BaseModel
import urllib3
import certifi

# Disable HTTPS verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

class EnhancedRetriever(BaseRetriever, BaseModel):
    """Enhanced retriever with fuzzy matching capabilities."""
    
    vectorstore: Any = Field(description="Vector store for similarity search")
    documents: List[Document] = Field(description="List of documents")
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def from_components(cls, vectorstore: Any, documents: List[Document]) -> "EnhancedRetriever":
        """Create an instance from components."""
        return cls(vectorstore=vectorstore, documents=documents)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Perform fuzzy matching to find similar questions
        all_texts = [doc.page_content for doc in self.documents]
        matches = process.extract(query, all_texts, scorer=fuzz.token_sort_ratio)
        
        # Get the best match if it's above threshold (80%)
        best_matches = [match for match in matches if match[1] > 80]
        
        if best_matches:
            # If we found similar questions, use them to enhance the search
            enhanced_query = best_matches[0][0]
            st.info(f"📝 Searching for similar context to: '{enhanced_query}'")
        else:
            enhanced_query = query
        
        # Perform vector similarity search
        return self.vectorstore.similarity_search(enhanced_query, k=4)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

def load_db(documents: list, chain_type: str = "stuff", k: int = 4):
    """Create QA chain from documents with enhanced similarity search."""
    try:
        # Get LLM instance from config
        llm = get_llm()
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Create enhanced retriever using the factory method
        retriever = EnhancedRetriever.from_components(vectorstore, documents)
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

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
        3. If the question requires basic analysis or calculations:
           - Perform the necessary calculations or analysis
           - Clearly explain the steps and results
           - Start your response with "🦉 Based on my analysis:"
        4. Always aim to be helpful and accurate
        5. If you're unsure, be honest about it

        Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # Create the chain using the enhanced retriever
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

def get_video_id(url):
    """Extract video ID from YouTube URL."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path[1:]
        if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            return parse_qs(parsed_url.query)['v'][0]
    except Exception:
        return None
    return None

def get_youtube_title(url, retries=3):
    """Get YouTube video title with retries."""
    for attempt in range(retries):
        try:
            yt = YouTube(url)
            return yt.title
        except Exception as e:
            if attempt == retries - 1:
                st.warning(f"Could not get video title: {str(e)}")
                return None
            time.sleep(1)  # Wait before retrying

def download_youtube_audio(url, retries=3):
    """Download YouTube audio with retries."""
    for attempt in range(retries):
        try:
            yt = YouTube(url)
            audio_stream = yt.streams.filter(
                only_audio=True,
                file_extension='mp4'
            ).first()
            
            if not audio_stream:
                raise Exception("No audio stream available")
                
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            audio_stream.download(filename=temp_audio_path)
            return temp_audio_path, yt.title
            
        except Exception as e:
            if attempt == retries - 1:
                raise Exception(f"Failed to download audio after {retries} attempts: {str(e)}")
            time.sleep(1)  # Wait before retrying

def process_document(file_path: str, file_type: str):
    """Process document with chunking."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        if file_type == 'youtube':
            video_id = get_video_id(file_path)
            if not video_id:
                st.error("Invalid YouTube URL")
                return False
                
            try:
                # First try YouTube's built-in transcripts
                transcript_text = None
                yt_title = None
                
                try:
                    # Try multiple languages if English isn't available
                    languages_to_try = ['en', 'en-US', 'en-GB', 'a.en']
                    transcript = None
                    
                    for lang in languages_to_try:
                        try:
                            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                            break
                        except Exception:
                            continue
                    
                    if transcript:
                        transcript_text = ' '.join([entry['text'] for entry in transcript])
                        yt_title = get_youtube_title(file_path)
                    else:
                        raise Exception("No transcript available in any supported language")
                    
                except Exception as transcript_error:
                    st.info("Built-in transcript not available. Attempting to transcribe audio...")
                    
                    # Fallback to audio download and transcription
                    temp_audio_path = None
                    try:
                        # Download audio
                        with st.spinner("Downloading audio..."):
                            temp_audio_path, yt_title = download_youtube_audio(file_path)
                            
                            # Add a small delay to ensure file is written
                            time.sleep(1)
                            
                            # Transcribe using Whisper
                            with st.spinner("Transcribing audio..."):
                                transcript_text = transcribe_video(temp_audio_path)
                                
                    except Exception as download_error:
                        raise Exception(f"Audio download/transcription failed: {str(download_error)}")
                    finally:
                        # Clean up temporary file
                        if temp_audio_path and os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)
                
                if not transcript_text:
                    st.error("Failed to get transcript from both methods")
                    return False
                
                # Use video title if available, otherwise use ID
                source_key = f"YouTube: 📺 {yt_title or f'Video {video_id}'}"
                
                # Split transcript into chunks
                texts = text_splitter.split_text(transcript_text)
                documents = [Document(page_content=t, metadata={"source": source_key}) for t in texts]
                
            except Exception as e:
                st.error(f"Could not process video: {str(e)}")
                handle_error("Error processing YouTube video", e)
                return False
                
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

