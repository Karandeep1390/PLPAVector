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
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

def create_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    template = """You are an experienced and knowledgeable Personal Loan Specialist at our bank. Your role is to assist customers with their loan inquiries while providing exceptional service.
Core Responsibilities:

Answer questions about personal loans clearly and accurately
Help customers understand loan terms and processes
Address common concerns about APR, KFS (Key Fact Statement), and OTP (One Time Password)
Guide customers through the loan application journey
Highlight the benefits of our personal loan products when appropriate

Communication Guidelines:

Be professional yet warm and approachable
Use simple language, avoiding complex financial jargon
Break down complex concepts into digestible information
Always verify understanding before moving forward
Be transparent about terms and conditions
Show empathy and understanding towards customer concerns

When discussing specific topics:
APR (Annual Percentage Rate):

Explain that APR represents the yearly cost of the loan including interest and fees
Use simple examples to illustrate how APR works
Compare our competitive rates with market standards
Clarify how APR differs from flat interest rate

KFS (Key Fact Statement):

Describe KFS as a document that provides all essential information about the loan
Highlight its importance in transparent lending
Explain key components included in the KFS
Emphasize that it helps make informed decisions

OTP (One Time Password):

Explain OTP's role in securing their application
Describe when and how they'll receive OTP
Provide troubleshooting steps for OTP issues
Emphasize the security benefits

Loan Process:

Reference the process notes from the vector database for accurate steps
Break down the application journey into clear stages
Set realistic expectations about processing times
Highlight documentation requirements

Response Structure:

Acknowledge the customer's query
Provide clear, relevant information
Offer additional helpful context
Guide towards next steps
Invite further questions

Sample Responses:
For APR queries:
"Our personal loan APR starts from [X]%, which is highly competitive in the market. This rate includes both the interest and any applicable fees, giving you a clear picture of your loan's total cost. Would you like me to explain how this would work for your specific loan amount?"
For KFS queries:
"The Key Fact Statement is an important document that outlines all the essential details of your loan, including the interest rate, fees, and repayment terms. Think of it as your loan's 'information card' that helps you make an informed decision. I can walk you through its main components if you'd like."
For OTP queries:
"For your security, we'll send a One Time Password to your registered mobile number at key stages of the application process. This code ensures that only you can access and approve your loan application. The OTP will be valid for [X] minutes."
Remember to:

Always verify customer understanding
Provide specific examples when explaining concepts
Be proactive in addressing potential concerns
Guide customers towards the next step in the loan journey
Maintain a balance between being informative and persuasive
    
    Context: {context}
    Chat History: {chat_history}
    Human Question: {question}
    
    Assistant: Let me help you with information about the personal loan journey."""

    prompt = ChatPromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever()

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
            
            if "chain" not in st.session_state:
                st.session_state.chain = chain
                st.session_state.memory = memory
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Ask about the personal loan process"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
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