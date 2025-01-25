# FileWhispererApp🦉
FileWhisperer is an intelligent document assistant that transforms how you interact with your documents, videos, and web content. Using advanced AI technology, it allows you to have natural conversations with your content, making information retrieval intuitive and efficient.

## 🌟 Features

- **Multi-Document Chat**: Chat with one or multiple documents simultaneously
- **Multiple Format Support**:
  - PDF documents (`.pdf`)
  - Word documents (`.docx`)
  - PowerPoint presentations (`.pptx`)
  - Excel spreadsheets (`.xlsx`)
  - CSV files (`.csv`)
  - Text files (`.txt`)
  - YouTube video transcripts
  - Web pages

- **Smart Document Handling**:
  - Automatic title extraction for web pages and YouTube videos
  - Visual indicators for different document types
  - Easy switching between single and multiple document modes

## 🚀 Streamlit Cloud Deployment

1. Fork this repository to your GitHub account

2. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud):
   - Connect your GitHub account
   - Select the forked repository
   - Select the main file: `streamlit_app.py`

3. Add your secrets in Streamlit Cloud:
   - Go to your app settings
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "sk-your-key-here"
     ```

## 💻 Local Development

1. Clone the repository:
