# Langchain Ask Guideline (Tutorial)

This is a Python application that allows you to load a Guideline PDF and ask questions about it using natural language. The application uses a LLM to generate a response about your Guideline PDF. The LLM will not answer questions unrelated to the document.

## How it works

The application reads the Guideline PDF and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response.

The application uses Streamlit to create the GUI and Langchain to deal with the LLM.

## Installation

To install the repository, please clone this repository and install the requirements:

```
pip install -r requirements.txt
```

You will also need to add your OpenAI API key to the `.env` file.

## Usage

To use the application, run the `main.py` file with the streamlit CLI (after having installed streamlit):

```
streamlit run app.py
```

## Cache

The application uses a cache to store the embeddings of the chunks. This is to avoid having to recompute the embeddings every time the user asks a question. The cache is stored in the `cache` folder. If you want to clear the cache, you can delete the contents of the `cache` folder.

It also uses FAISS to speed up the search for similar chunks. The FAISS index is stored in the `knowledge_base` folder. If you want to clear the FAISS index, you can delete the contents of the `knowledge_base` folder. The index will be reused for the same document.
