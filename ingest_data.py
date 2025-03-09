from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

def load_and_process_files(data_dir: str):
    """Load PDFs and markdown files from directory and split into chunks."""
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    md_loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=lambda path: TextLoader(path, encoding='utf-8')
    )

    pdf_documents = pdf_loader.load()
    md_documents = md_loader.load()
    documents = pdf_documents + md_documents

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_dir: str):
    """Create and persist a Chroma vector store."""
    # Remove existing vector store if present
    if os.path.exists(persist_dir):
        print(f"Removing existing vector store from {persist_dir}")
        shutil.rmtree(persist_dir)

    # Initialize the HuggingFace embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Build and persist the new Chroma vector store
    print("Building and saving the new vector store...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    return vector_db

def main():
    # Define paths for input data and vector database
    input_data_dir = os.path.join("data")
    vector_db_dir = os.path.join("chroma_db")

    # Load PDFs and markdown files and break them into smaller chunks
    print("Starting file processing...")
    document_chunks = load_and_process_files(input_data_dir)
    print(f"Generated {len(document_chunks)} document chunks from files")

    # Build and store the vector database
    print("Building the vector store...")
    vector_db = create_vector_store(document_chunks, vector_db_dir)
    print(f"Vector store successfully created and saved at {vector_db_dir}")

if __name__ == "__main__":
    main()
