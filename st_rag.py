import os 
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF; /* Light background */
        color: #000000; /* Dark text */
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #F0F0F0 !important; /* Light background */
        color: #000000 !important; /* Dark text */
        border: 1px solid #CCCCCC !important; /* Lighter border */
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #F0F0F0 !important; /* Light background */
        border: 1px solid #DDDDDD !important; /* Light border */
        color: #333333 !important; /* Dark text */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #E5E5E5 !important; /* Slightly darker light background */
        border: 1px solid #DDDDDD !important; /* Light border */
        color: #333333 !important; /* Dark text */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important; /* Keep the avatar color */
        color: #000000 !important; /* Dark avatar text */
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #000000 !important; /* Dark text */
    }
    
    .stFileUploader {
        background-color: #F0F0F0; /* Light background */
        border: 1px solid #CCCCCC; /* Light border */
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important; /* Keep heading color */
    }
    </style>
    """, unsafe_allow_html=True)


PROMPT_TEMPLATE = """
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge,
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the provided knowledge. Use the provided context to answer the query. 
        If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

 
CO_API_KEY=os.environ['COHERE_API_KEY']
LANGUAGE_MODEL= ChatCohere(temperature=0.5, model="command-r-plus")

PDF_STORAGE_PATH = 'document_store/pdfs/'
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

from langchain_huggingface import HuggingFaceEmbeddings
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data
def save_uploaded_file(_uploaded_file):
    file_path = PDF_STORAGE_PATH + _uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(_uploaded_file.getbuffer())
    return file_path

@st.cache_data
def load_pdf_documents(_file_path):
    document_loader = PDFPlumberLoader(_file_path)
    return document_loader.load()

@st.cache_data
def chunk_documents(_raw_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(_raw_documents)
    
    return chunks

CHROMA_PATH = r"chroma_db"
vector_store = Chroma(
    collection_name="document_collection",
    embedding_function=EMBEDDING_MODEL,
    persist_directory=CHROMA_PATH,
)

@st.cache_data
def index_documents(_document_chunks):

    uuids = [str(uuid4()) for _ in range(len(_document_chunks))]

    vector_store.add_documents(documents=_document_chunks, ids=uuids)

@st.cache_data
def find_related_documents(_query):
    return vector_store.similarity_search(_query)


def generate_answer(_user_query, _context_documents):
    num_results = 5
    retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

    docs = retriever.invoke(_user_query, {"document_context": _context_documents})

    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"

    if _user_query is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge,
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the provided knowledge.

        The question: {_user_query}

        The knowledge: {knowledge}

        """

        for response in LANGUAGE_MODEL.stream(rag_prompt):
            partial_message += response.content
        
    return partial_message


st.title("📘 RAG Agent")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")


uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions to the LLM.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)