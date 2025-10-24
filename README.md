# RAG-based Q&A Chatbot

## Overview

An AI-powered chatbot built using LangChain, Groq LLM, Ollama and FAISS, which enables you to upload documents or URLs and ask context-aware questions.
The chatbot answers only based on the uploaded document and refuses to answer unrelated questions.

------------------------------------------------------------------------

## Features

- Upload or link a document — Supports PDF, DOCX, and TXT files, or a valid website URL.
- Retrieval-Augmented Generation (RAG) — Answers are generated only from your document content.
- Conversational Memory — Maintains context through ongoing chat history.
- Instant Reset — Clear the chat and reload a new document easily.

------------------------------------------------------------------------

## Tech Stack

| Component                  | Technology                |
| -------------------------- | ------------------------- |
| **Frontend/UI**            | Streamlit                 |
| **Language Model**         | Groq Llama-3.1-8B-Instant |
| **Embeddings**             | Ollama EmbeddingGemma     |
| **Vector Store**           | FAISS                     |
| **Framework**              | LangChain                 |
| **Environment Management** | Python Dotenv             |

------------------------------------------------------------------------

## Project Structure

``` bash
📘 RAG-QA-Chatbot/
│
├── main.py                 # Main Streamlit app
├── document_processing.py  # Handles document loading and splitting
├── utils.py                # Helper utilities 
├── .env                    # Store your GROQ_API_KEY
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

------------------------------------------------------------------------

## How to Run Locally

-  Step 1: Clone the Repository

    ``` bash
    git clone https://github.com/naja-an/RAG-Based-Q-A-Chatbot.git
    cd rag-based-q-a-chatbot

    ```

-  Step 2: Create a virtual environment
    ``` bash
    python -m venv venv
    source venv/bin/activate      # (Linux/Mac)
    venv\Scripts\activate         # (Windows)
    ```

-  Step 3: Install Dependencies

    If using uv:

    ``` bash
    uv sync

    ```
    Or using pip:

    ``` bash
    pip install -r requirements.txt

    ```

-  Step 4: Set your Groq API Key and download embeddinggemma model from ollama
    Create a .env file in the project root:
    ``` bash
    GROQ_API_KEY=your_api_key_here
    ```
    To download embedding model:
    ``` bash
    ollama run embeddinggemma
    ```

-  Step 5: Run the Streamlit App
    ``` bash
    streamlit run main.py
    # For uv users:
    uv run streamlit run main.py
    ```

-  Step 6: Test the App

    Once running, open the local Streamlit URL (e.g., http://localhost:8501), upload a file or paste a link and start querying.

------------------------------------------------------------------------

## Project Workflow
1. Upload a document or enter a URL
The user uploads a document (PDF, DOCX, or TXT) or provides a URL.
The content is extracted and split into smaller chunks using a RecursiveCharacterTextSplitter for efficient retrieval.

2. Embedding & Vector Store Creation
Each chunk is converted into embeddings using Ollama Embeddings (embeddinggemma model).
These embeddings are stored in a FAISS vector store for fast similarity search.

3. User Query
Context-Aware Question Answering
For every user query, a history-aware retriever reformulates the question based on previous chat context (e.g., “What is Bert?” (first question) 
“What is it used for?” (second question) is converted to “What is BERT used for?”).
The refined query retrieves relevant chunks from FAISS, and the LLM generates a concise, contextually accurate response.

4. Conversation Memory & Continuity
Chat history is maintained using ConversationBufferMemory, allowing for continuous, context-aware dialogue across multiple queries within a session.

------------------------------------------------------------------------

