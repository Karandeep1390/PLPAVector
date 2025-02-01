import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from docx import Document
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if "messages" not in st.session_state:
    st.session_state.messages = []

def read_docx(file_path):
    """Extract text from DOCX file."""
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

def create_vectorstore(text_chunks):
    """Create FAISS vectorstore from text chunks using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def split_text(text):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_chain(vectorstore):
    """Create the RAG chain with conversation memory."""
    llm = ChatOpenAI(temperature=0.1)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    template = """You are a call center assistant for kotak bank’s personal loans journey. 
    Your job is to convince customer to take loan from Kotak by highlighting the benefits of Kotak. 
    For example, low interest rates, trust with Kotak to convince the customer. 
    Additionally, you should also mention about the limited time offer to create urgency 
    to the customer to take loan and you should be helping the customer out in clarifying his doubts or questions.
    Your answer should be short and precise. Try to answer it from context
    
    Context: {context}
    Chat History: {chat_history}
    Human Question: {question}
    
    Assistant: Let me help you with information about the personal loan journey."""

    prompt = ChatPromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    chain = (
        {"context": retriever, 
         "chat_history": memory.load_memory_variables,
         "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain, memory

def main():
    st.title("Personal Loan Journey Assistant")
    docx_file_path = "process_note.docx" 
    
    if os.path.exists(docx_file_path):
        try:
            text = read_docx("process_note.docx")
            chunks = split_text(text)
            vectorstore = create_vectorstore(chunks)
            chain, memory = create_chain(vectorstore)
            
            # Chat interface
            if "chain" not in st.session_state:
                st.session_state.chain = chain
                st.session_state.memory = memory
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Accept user input
            if prompt := st.chat_input("Ask about the personal loan process"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    response = st.session_state.chain.invoke(prompt)
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    st.session_state.memory.save_context({"input": prompt}, {"output": response.content})
        
        except Exception as e:
            st.error(f"Error processing the document: {str(e)}")
    
    else:
        st.error(f"Document file not found at {docx_file_path}. Please check the file path.")

if __name__ == "__main__":
    main()