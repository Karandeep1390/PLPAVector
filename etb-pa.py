import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from docx import Document
from dotenv import load_dotenv
import os
import logging
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessNoteVectorStore:
    def __init__(self, chunk_size=500, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings()

    def read_docx(self, file_path):
        """Read and preprocess process note from DOCX file."""
        try:
            doc = Document(file_path)
            sections = []
            current_section = []

            # Read the text sections as before
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    # Detect new sections
                    if re.match(r'^\d+[\.\)]|^Step|^STEP|^[A-Z\s]{5,}:', text):
                        if current_section:
                            sections.append('\n'.join(current_section))
                            current_section = []
                    logger.info(f"Adding to section following text: {text}")
                    current_section.append(text)

            # Add the last section
            if current_section:
                sections.append('\n'.join(current_section))

            return sections
        except Exception as e:
            logger.error(f"Error reading process note: {str(e)}")
            raise

    def split_text(self, sections):
        """Split text into chunks while preserving process steps."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "? "]
            )

            chunks_with_metadata = []
            for i, section in enumerate(sections):
                chunks = text_splitter.split_text(section)
                for j, chunk in enumerate(chunks):
                    metadata = {
                        'section_id': i,
                        'chunk_id': j,
                        'source': f"Section {i + 1}, Chunk {j + 1}",
                        'content_length': len(chunk)
                    }
                    chunks_with_metadata.append((chunk, metadata))

            return chunks_with_metadata
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise

    def create_vectorstore(self, chunks_with_metadata):
        """Create vector store with process-aware metadata and images."""
        try:
            # Create text-based vectorstore
            texts = [chunk[0] for chunk in chunks_with_metadata]
            metadatas = [chunk[1] for chunk in chunks_with_metadata]
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )

            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

def create_chain(vectorstore):
    """Create the conversation chain with retrieval-based memory."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_template(
                """You are Ajinkya, a friendly and helpful chat assistant for Kotak Personal Loans customers. Your role is to assist customers with any questions or concerns about Kotak Bank's personal loans. You should:Respond based only on the information available in the knowledge base about Kotak Personal Loans. If you're unsure of something, admit that you don’t have the answer instead of providing incorrect information.
Use clear, simple, and precise language to ensure that your responses are easily understood by customers, regardless of their familiarity with banking terms.
Maintain a warm, empathetic, and friendly tone, especially when addressing customer concerns. Acknowledge the customer’s emotions when appropriate (e.g., if they’re frustrated or uncertain).
When troubleshooting or providing setup instructions, offer clear, step-by-step guidance. Make sure to check in with the customer to confirm they’re following along before moving forward.
Periodically ask the customer if they are satisfied with the assistance, and if they need further help, ensuring they feel heard and valued throughout the interaction.
Keep your responses at a steady, moderate pace, and ensure they are easy to follow.
Your ultimate goal is to make every customer feel heard, supported, and delighted with the service. Always strive to provide helpful, clear, and empathetic responses to make every interaction a positive experience.
                
                {context}
                Human Question: {question}"""
            )
        },
    )
    return chain, memory

def main():
    st.title("Personal Loan Process Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    docx_file_path = "process_note.docx"
    
    if os.path.exists(docx_file_path):
        try:
            # Initialize ProcessNoteVectorStore
            doc_processor = ProcessNoteVectorStore(
                chunk_size=1000,
                chunk_overlap=300
            )
            
            # Process document
            sections = doc_processor.read_docx(docx_file_path)
            chunks_with_metadata = doc_processor.split_text(sections)
            vectorstore = doc_processor.create_vectorstore(chunks_with_metadata)
            
            # Create chain if not in session state
            if "chain" not in st.session_state:
                chain, memory = create_chain(vectorstore)
                st.session_state.chain = chain
                st.session_state.memory = memory
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Handle new user input
            if prompt := st.chat_input("Ask about the loan process steps"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    chain = st.session_state.chain
                    response = chain({"question": prompt})
                    
                    # Get the answer from the response and save it in memory
                    answer = response["answer"]  # Only get the answer
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Save context in memory
                    st.session_state.memory.save_context({"input": prompt}, {"output": answer})
        
        except Exception as e:
            st.error(f"Error processing the document: {str(e)}")
            logger.error(f"Application error: {str(e)}")
    
    else:
        st.error(f"Document file not found at {docx_file_path}. Please check the file path.")

if __name__ == "__main__":
    main()
