# Langchain RAG

## RAG (Retriever-Augmented Generation) with Langchain, Cohere, Hugging Face Embeddings, and Chroma DB

This project implements a Retriever-Augmented Generation (RAG) pipeline using **Langchain**, **Cohere** (for language model), **Hugging Face Embeddings**, and **Chroma DB** (as the vector database). It demonstrates how to combine these tools to create a robust search and response system powered by language models.

## Overview

RAG is a powerful architecture that enhances the capabilities of language models by incorporating a retrieval component to retrieve relevant information from a knowledge base before generating an answer. This project leverages Langchain to tie everything together, while **Cohere** powers the language model, **Hugging Face** provides the embeddings, and **Chroma DB** serves as the vector database to store and search through embeddings.

## Technologies Used

- **Langchain**: Framework for chaining together language models, tools, and agents.
- **Cohere**: Language model provider for text generation tasks.
- **Hugging Face Embeddings**: Used to transform text into numerical representations (embeddings).
- **Chroma DB**: Vector database for storing and retrieving embeddings.

## Setup and Installation

### Prerequisites

1. Python 3.13
2. Install required Python libraries using `pip`.

### Install Dependencies

1. Clone the repository:

    ```bash
    git clone https://github.com//lumeirne/langchain_rag.git
    cd langchain_rag
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### API Keys and Configuration

1. **Cohere API Key**: Sign up for a Cohere API key at [https://cohere.ai/](https://cohere.ai/), and save it in the environment variables:

    ```bash
    export COHERE_API_KEY="your-cohere-api-key"
    ```

2. **Hugging Face Embeddings**: Install the Hugging Face library and set up your credentials:

    ```bash
    pip install langchain_huggingface
    ```

    Then, log in to Hugging Face:

    ```bash
    huggingface-cli login
    ```

3. **Chroma DB Configuration**: Ensure Chroma DB is correctly set up and initialized in your code. It handles the vector storage and retrieval.

## Running the Application

Once you have everything set up, you can run the application using the following command:

```bash
python -m streamlit run st_rag.py #For streamlit version
python gr_rag.py #For gradio version
