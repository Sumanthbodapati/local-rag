import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Define paths
data_directory = "data"

def build_rag_system():
    # Load documents
    print("Loading documents...")
    loader = DirectoryLoader(data_directory, glob='**/*.txt')
    documents = loader.load()

    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    print("Creating embeddings and FAISS vector store...")
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(texts, embeddings)

    # Build Retrieval-Augmented Generation system
    print("Building RAG system...")
    retriever = vectordb.as_retriever()
    qa_system = RetrievalQA(llm=OpenAI(), retriever=retriever)

    return qa_system

def main():
    # Ensure data directory exists
    if not os.path.exists(data_directory):
        print(f"Data directory '{data_directory}' does not exist. Please create it and add your documents.")
        return

    # Build the RAG system
    qa_system = build_rag_system()

    # Interact with the user
    print("\nRAG system is ready! Ask your questions below.")
    while True:
        question = input("\n> Enter your question: ")
        if question.lower() in ["exit", "quit"]:
            print("Exiting RAG system. Goodbye!")
            break

        print("Fetching context and generating answer...")
        answer = qa_system.run(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()