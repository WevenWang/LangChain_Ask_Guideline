from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
import os


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask New Wave Guidelines" )
    st.header("Ask New Wave Guidelines" )
    
    
    knowledge_base = load_knowledge_base()
    
    # show user input
    user_question = st.text_input("Ask a question about your New Wave Guidelines:")
    if user_question:
      docs = knowledge_base.similarity_search(user_question)
      
      llm = OpenAI()
      chain = load_qa_chain(llm, chain_type="stuff")
      with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
          
      st.write(response)


def load_knowledge_base():
    file_path = 'assets/new_wave_guideline.pdf'
    loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=os.getenv('AZURE_COGNITIVE_ENDPOINT'), api_key=os.getenv('AZURE_COGNITIVE_KEY'), file_path=file_path, api_model="prebuilt-layout"
    )
      
    documents = loader.load()

    # print(documents[0].page_content[:500]) 
    
    
    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
      separators=["\n", "|"],
      chunk_size=1000,
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
    try:
      knowledge_base = FAISS.load_local("knowledge_base", cached_embedder)
      print("Loaded knowledge base from local storage")
    except:
      knowledge_base = FAISS.from_documents(chunks, cached_embedder)
      knowledge_base.save_local("knowledge_base")
      print("Saved knowledge base to local storage")
    return knowledge_base


if __name__ == '__main__':
    main()
