import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from document_processing import DocumentLoaderException, create_docs
from utils import  rag_prompt, contextualize_query_prompt

# ===============================
# ENVIRONMENT & MODEL SETUP
# ===============================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

embeddings = OllamaEmbeddings(
    model="embeddinggemma",
)
llm = ChatGroq(model_name="llama-3.1-8b-instant")

# ===============================
# STREAMLIT UI
# ===============================
st.title("📘 RAG-based Q&A Chatbot")
st.write("Upload a document or a URL and ask any query related to it.")

uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
st.write("----------- OR -----------")
url = st.text_input("Enter a valid URL")

# ===============================
# RESET BUTTON
# ===============================
if st.button("🔄 Reset Chat / Start Over"):
    for key in ["memory", "vectorstore", "retrieval_chain"]:
        st.session_state.pop(key, None)
    uploaded_file = None
    url = None
    st.rerun()

# ===============================
# SESSION STATE INITIALIZATION
# ===============================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None


# ===============================
# LOAD AND PROCESS DOCUMENT
# ===============================
if (uploaded_file or url) and st.session_state.vectorstore is None:
    with st.spinner("📄 Processing document..."):
        # Load and create documents
        try:
            docs = create_docs(uploaded_file,url)
        except DocumentLoaderException as e:
            print(e)
            st.error(e)
            st.stop()
        except Exception as e:
            print(e)
            st.error("Sorry, there was some error while loading the contents of your document")
            st.stop()

        # Create FAISS vectorstore
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

        # Create retriever 
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        
        # Create a history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=st.session_state.retriever,
            prompt=contextualize_query_prompt
        )

        # Create a document chain
        document_chain = create_stuff_documents_chain(llm, rag_prompt)

        # Create a Retrieval chain
        st.session_state.retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )

    st.success("✅ Document loaded successfully! You can now ask questions.")

# ===============================
# USER QUERY HANDLING
# ===============================
if st.session_state.retrieval_chain:

    if st.session_state.memory and st.session_state.memory.chat_memory.messages:
        for msg in st.session_state.memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "Assistant"
            with st.chat_message(role.lower()):
                st.markdown(msg.content)

    user_query = st.chat_input("Ask something about the document...")

    if user_query:
        # Immediately display the user's message
        with st.chat_message("human"):
            st.markdown(user_query)

        with st.spinner("🤔 Thinking..."):
            response = st.session_state.retrieval_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.memory.chat_memory.messages
            })
            answer = response["answer"]
        # Display assistant's response
        with st.chat_message("assistant"):
            st.markdown(answer)
        # Update memory
        st.session_state.memory.chat_memory.add_user_message(user_query)
        st.session_state.memory.chat_memory.add_ai_message(answer)


