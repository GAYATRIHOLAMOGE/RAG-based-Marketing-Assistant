"""
Document ingestion pipeline.
Loads text files from ./data, splits them into chunks,
embeds with a local sentence-transformer model, and stores in ChromaDB.
"""

import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents(data_dir: str):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from {data_dir}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n---\n\n", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks, chroma_dir: str):
    if Path(chroma_dir).exists():
        shutil.rmtree(chroma_dir)
        print(f"Cleared existing vector store at {chroma_dir}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding chunks and building ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    print(f"Vector store saved to {chroma_dir} ({vectorstore._collection.count()} vectors)")
    return vectorstore


def main():
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found. Add .txt files to the ./data folder and re-run.")
        return
    chunks = split_documents(docs)
    build_vector_store(chunks, CHROMA_DIR)
    print("\nIngestion complete. Run 'streamlit run app.py' to start the chatbot.")


if __name__ == "__main__":
    main()
