from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA, ConversationalRetrievalChain

from utils import load_knowledge_base




def main():
    load_dotenv()
    st.set_page_config(page_title="Ask New Wave Guidelines" )
    st.header("Ask New Wave Guidelines" )
    
    
    knowledge_base = load_knowledge_base()
    st.session_state.retriever = knowledge_base.as_retriever()
    
    # show user input
    # user_question = st.text_input("Ask a question about your New Wave Guidelines:")
    # if user_question:
    #   docs = knowledge_base.similarity_search(user_question)
      
    #   llm = OpenAI()
    #   chain = load_qa_chain(llm, chain_type="stuff")
    #   with get_openai_callback() as cb:
    #     response = chain.run(input_documents=docs, question=user_question)
    #     print(cb)
          
    #   st.write(response)

    # convert above to a chatbot with 
  
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result
   


if __name__ == '__main__':
    main()
