from typing import Annotated, TypedDict
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from app.ragservices.loaders import DocumentLoader
from app.ragservices.chunker import Chunker
from app.ragservices.embed_store import VectorStore

from app.ragservices.reranker import ReRanker
from app.schemas.llms import InitiateLLMs
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GRADE_MODEL = "openai/gpt-4o-mini"
GENERATION_MODEL = "openai/gpt-4o-mini"
PERSIST_DIR = "/workspaces/rag/app/chroma"

shared_embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
store = VectorStore(embedding=shared_embedding, persist_dir=PERSIST_DIR)
reranker = ReRanker(RERANKER_MODEL)
llm = InitiateLLMs(GENERATION_MODEL, GRADE_MODEL)

class StoringRAGStateManagment(TypedDict):
    documents: list[Document]
    chunks: list[Document]

@traceable(name="Load Documents")
def load_documents(state: StoringRAGStateManagment) -> dict:

    new_documents = []

    data_files_path = "/workspaces/rag/app/data"
    for root, _, filenames in os.walk(data_files_path):
        for filename in filenames:
            file_path = os.path.join(root,filename)
            ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            loader = DocumentLoader(ext,file_path)
            documents = loader.choosing_loader()
            new_documents.extend(documents)

    return {
        "documents" : new_documents
    }

@traceable(name="Split Documents")
def chunk_documents(state: StoringRAGStateManagment) -> dict:

    chunker = Chunker(500,200)
    splitter = chunker.split(state["documents"])
    return {
        "chunks" : splitter
    }

@traceable(name="Storing Documents")
def embedding_storing(state: StoringRAGStateManagment) -> dict:

    store.build_store(state["chunks"])
    return {}


graph1 = StateGraph(StoringRAGStateManagment)
graph1.add_node("Loading", load_documents)
graph1.add_node("Chunking", chunk_documents)
graph1.add_node("Storing",embedding_storing)

graph1.add_edge(START,"Loading")
graph1.add_edge("Loading", "Chunking")
graph1.add_edge("Chunking", "Storing")
graph1.add_edge("Storing",END)

ingestion_app = graph1.compile()