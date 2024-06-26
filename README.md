﻿# Conversational-Document-Analysis-and-Retrieval-System


Developed a Streamlit application leveraging natural language processing techniques to enable conversational interaction with multiple documents (PDFs). The system allows users to ask questions about uploaded documents and retrieves relevant information from the text.

 **Technologies Used**:
   - Streamlit: Built the user interface for the application.
   - PyPDF2: Extracted text from PDF documents.
   - Langchain: Utilized for text splitting into manageable chunks and generating embeddings.
   - OpenAI: Integrated OpenAI's language model for conversational responses.
   - FAISS: Implemented a local database for storing numerical representations of text chunks.
   - dotenv: Handled environment variables, ensuring secure API key management.

**Functionality**:
   - Uploaded PDF documents are processed to extract text, which is then split into chunks for efficient handling.
   - Text chunks are embedded using OpenAI's embeddings and stored in a local database for quick retrieval.
   - The application supports conversational interaction, allowing users to ask questions about the documents.
   - Utilized a conversational retrieval chain with memory to facilitate context-aware responses and maintain conversation history.

**Features**:
   - Supports uploading multiple PDF documents for comprehensive document analysis.
   - Dynamically updates conversation history, allowing for seamless follow-up questions and context retention.
   - Utilizes an AI-powered language model to provide intelligent responses to user queries.

**Impact**:
   - Enhanced document accessibility and analysis through natural language interaction, catering to users who prefer conversational interfaces.
   - Streamlined document retrieval and information extraction processes, improving efficiency in document-based tasks.
   - Demonstrated proficiency in leveraging cutting-edge NLP techniques and technologies to develop practical applications.
