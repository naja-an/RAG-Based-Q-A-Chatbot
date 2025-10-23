import re
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def is_valid_url(url):
    """
    Validates a URL string using a regular expression.
    Args:
        url: string representing a URL
    Returns:
        A boolean value (True/False)
    """
    url_pattern = re.compile(
        r'^(https?://)?'  # Optional http or https scheme
        r'(www.)?'       # Optional 'www.'
        r'([a-zA-Z0-9-]+.)+' # One or more subdomains/domain parts
        r'([a-zA-Z]{2,6})' # Top-level domain (TLD)
        r'(/[\w\-.?%@#&=]*)*' # Optional path, query, and fragment
        r'$'
    )
    return bool(url_pattern.fullmatch(url))


def is_query_relevant_llm(query, llm, docs_summary):
    """LLM-based relevance check.
    Args:
        query: A string representing the user query.
        llm: A chat model, any child class of BaseChatModel
        docs_summary: A string containing the summary of the user provided document.
    Returns:
        A boolean value (True/False)
    """
    guard_prompt = f"""
    You are a relevance classifier. Determine if the user's question is related to the document.
    Respond with ONLY "YES" or "NO".

    Question: {query}
    Document summary: {docs_summary}
    """
    decision = llm.invoke(guard_prompt).content.strip().lower()
    return "yes" in decision

    
def summarize_docs(llm, retriever):
    """Generates a summary of the uploaded document.
    Args:
        llm: A chat model, any child class of BaseChatModel
        retriever: VectorStoreRetriever
    Returns: 
        A summary of the uploaded document.
    """
    prompt_template = """Write a concise summary of the following: "{context}" CONCISE SUMMARY: """
    prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever, chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
    )
    
    summary = qa_chain.invoke("Please summarize this document.")['result']
    return summary

rag_prompt = ChatPromptTemplate.from_template("""
        Answer the user's question **only** based on the context below.
        If the answer is not in the document, say "Sorry, I couldn’t find that in the document."

        <context>
        {context}
        </context>

        Question: {input}
        """)

