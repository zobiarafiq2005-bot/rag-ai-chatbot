import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStoreManager:
    """Manages embedding generation and ChromaDB vector database storage.

    Keeps a single live Chroma instance (self.vector_store) in memory so
    documents ingested across multiple uploads all land in the same
    collection, and so a retriever is always available to query against.
    """

    def __init__(self, persist_directory: str = "vector_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        print(f"[INFO] Initializing Embedding Model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store: Optional[Chroma] = None

        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            try:
                self.load_vector_store()
            except Exception as e:
                print(f"[WARN] Could not load existing vector store: {e}")

    def create_vector_store(self, chunks: List[Document]) -> Chroma:
        """Embeds document chunks and saves them into local ChromaDB."""
        print(f"[INFO] Indexing {len(chunks)} chunks into ChromaDB at '{self.persist_directory}'...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("[SUCCESS] Vector Store created and persisted successfully!")
        return self.vector_store

    def add_documents(self, chunks: List[Document]) -> None:
        """Adds more chunks to the existing vector store, creating it first
        if this is the very first upload — lets multiple uploaded documents
        accumulate into one queryable collection."""
        if self.vector_store is None:
            self.create_vector_store(chunks)
        else:
            print(f"[INFO] Adding {len(chunks)} more chunks to existing ChromaDB...")
            self.vector_store.add_documents(chunks)
            print("[SUCCESS] Chunks added to existing Vector Store!")

    def load_vector_store(self) -> Chroma:
        """Loads an existing ChromaDB vector database from disk."""
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"Vector DB directory '{self.persist_directory}' does not exist yet.")

        print(f"[INFO] Loading existing ChromaDB from '{self.persist_directory}'...")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vector_store

    def as_retriever(self, **kwargs):
        """Convenience passthrough for the RAG chain. Raises a clear error
        if nothing has been indexed yet."""
        if self.vector_store is None:
            raise ValueError("Vector store is empty — ingest a document before querying.")
        return self.vector_store.as_retriever(**kwargs)