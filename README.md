# RAG-based Q&A Chatbot

## Overview

An AI-powered chatbot built using LangChain, Groq LLM, Ollama and FAISS, which enables you to upload documents or URLs and ask context-aware questions.
The chatbot answers only based on the uploaded document and refuses unrelated questions using a built-in LLM guardrail.

------------------------------------------------------------------------

## Features

- Upload or link a document — Supports PDF, DOCX, and TXT files, or a valid website URL.
- Retrieval-Augmented Generation (RAG) — Answers are generated only from your document content.
- LLM Guardrail — Detects and blocks irrelevant or out-of-scope questions.
- Conversational Memory — Maintains context through ongoing chat history.
- Instant Reset — Clear the chat and reload a new document easily.
- Document Summarization — Creates an automatic summary for guardrail decision-making.

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
├── utils.py                # Helper utilities (guardrail, summary, prompts)
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
The file is loaded and split into smaller chunks using a RecursiveCharacterTextSplitter.

2. Vectorization
Each chunk is converted into embeddings using Ollama Embeddings and stored in FAISS.

3. Document Summary
The system creates a quick summary of the content — used later by the guardrail.

4. User Query & Guardrail
Every user query is checked for relevance using an LLM-based filter (is_query_relevant_llm).

    - If irrelevant, the bot warns the user.

    - If relevant, it retrieves matching chunks from FAISS and generates an answer using ConversationalRetrievalChain.

5. Conversation Memory
Maintains the chat history using ConversationBufferMemory.

------------------------------------------------------------------------

