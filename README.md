# local-rag

This repository contains a Retrieval-Augmented Generation (RAG) system implemented in Python. The RAG system ingests a collection of documents, builds a retriever, and answers user questions by retrieving relevant context from the document set.

## Features
- Load and preprocess a collection of documents.
- Build a retriever to fetch relevant documents.
- Use a language model to generate answers based on the retrieved context.

## Requirements
Make sure you have the following libraries installed:
- `langchain`
- `openai`
- `faiss-cpu`

You can install the necessary libraries using the following command:
```
pip install langchain openai faiss-cpu
```

## Usage
To run the system, execute the Python script:
```
python rag_system.py
```

## Files
- `rag_system.py`: The main script for the RAG system.
- `data/`: Directory containing the documents for the RAG system.

## Example
1. Place your documents in the `data/` directory.
2. Run the script and ask questions interactively:
```
python rag_system.py
```

> Enter your question: What is the purpose of life?
> Fetching context...
> Model's Answer: The purpose of life is...

## License
This project is licensed under the MIT License.