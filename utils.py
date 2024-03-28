
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from langchain.document_loaders import PyMuPDFLoader
import streamlit

@streamlit.cache_resource
def load_knowledge_base():
    file_paths = ['assets/new_wave_guideline.pdf', 'assets/FHA_Handbook.pdf']
    documents = []  # list of Document objects
    for file_path in file_paths:
        # load the document
      loader = PyMuPDFLoader(file_path)
      documents.extend(loader.load())
      
    # print(documents[0].page_content[:500]) 

    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
      separators=["\n\n", "\n", " ",  ""],
      chunk_size=3000,
      chunk_overlap=200,
      length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # create embeddings
    embeddings = OpenAIEmbeddings()


    store = LocalFileStore("./cache/")


    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )
    # check if knowledge base is saved locally
    
    knowledge_base = FAISS.from_documents(chunks, cached_embedder)
      
    return knowledge_base