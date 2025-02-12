import os
import gradio as gr
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


CHROMA_PATH = r"chroma_db"
DATA_PATH = r"data"

loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)

uuids = [str(uuid4()) for _ in range(len(chunks))]

vector_store = Chroma(
    collection_name="document_collection",
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH,
)
vector_store.add_documents(documents=chunks, ids=uuids)


 
CO_API_KEY=os.environ['COHERE_API_KEY']

llm = ChatCohere(temperature=0.5, model="command-r-plus")

num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

def stream_response(message, history):

    docs = retriever.invoke(message)

    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"

    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge,
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the provided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(placeholder="Ask the LLM...",
        container=False,
        autoscroll=True,
        scale=7
        ),
)

chatbot.launch(share=True, debug=True)