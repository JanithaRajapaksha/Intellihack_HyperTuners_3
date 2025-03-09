from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from reasoning_model import reasoner  # Import reasoner from reasoning_model.py
import os

# Initialize vector store and embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

# Directory for the vector database
db_dir = os.path.join("chroma_db")

# Initialize the vector store
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)


def preview_documents(user_query: str, k: int = 5):
    """Fetch and display document chunks from the vector store."""
    docs = vectordb.similarity_search(user_query, k=k)
    for idx, doc in enumerate(docs, 1):
        print(f"Document {idx}:")
        print(doc.page_content)
        print('-' * 50)


def rag_with_reasoner(user_query: str) -> str:
    """
    Searches vector database for relevant context and generates a response using a reasoning model.
    These databases include information about the DeepSeek model.

    Args:
        user_query: The user's question.
    """
    # Retrieve relevant documents
    docs = vectordb.similarity_search(user_query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Create prompt for the reasoning model
    prompt = f"""Based on the following context, answer the user's question concisely.
    These databases include information about the DeepSeek model.
    If insufficient information is found, suggest a better query for RAG.

Context:
{context}

Question: {user_query}

Answer:"""

    # Generate response using reasoner (DeepSeek-R1)
    response = reasoner.run(prompt, reset=False)
    return response

if __name__ == "__main__":
    query = input("Enter your query: ")
    print("\nPreviewing retrieved documents:")
    preview_documents(query)
    print("\nGenerating response:")
    response = rag_with_reasoner(query)
    print(response)
