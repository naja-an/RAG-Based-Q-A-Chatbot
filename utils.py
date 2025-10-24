import re
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    

rag_prompt = ChatPromptTemplate.from_template("""
        Answer the user's question **only** based on the context below.
        If the answer is not in the document, say "Sorry, I couldn’t find that in the document."

        <context>
        {context}
        </context>

        Question: {input}
        """)

contextualize_query_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given the chat history and the latest user question, "
        "reformulate the question into a standalone query that includes context "
        "from the conversation if necessary. Do NOT answer the question, "
        "just rewrite it clearly if needed.")
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])