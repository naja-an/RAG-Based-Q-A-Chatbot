from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import is_valid_url



class DocumentLoaderException(Exception):
     def __init__(self, message):
        super().__init__(message)
        self.message = message

     def __str__(self):
           return self.message




def create_docs(uploaded_file, url):
    loader = None
    if url:
        if not is_valid_url(url):
              raise DocumentLoaderException(f"{url} is not valid.")
        loader = WebBaseLoader(url)
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        temp_path = f"temp.{file_extension}"
        with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        if file_extension == "pdf":
                loader = PyMuPDFLoader(temp_path)
        elif file_extension == "txt":
                loader = TextLoader(temp_path)
        elif file_extension == "docx":
                loader = UnstructuredWordDocumentLoader(temp_path)
        else:
                raise DocumentLoaderException("File type not supported")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_docs = text_splitter.split_documents(docs)

    return final_docs





