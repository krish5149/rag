from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorStore:

    def __init__(self, embedding: Optional[HuggingFaceEmbeddings] = None, embedding_model: str | None = None, persist_dir: str = "/workspaces/rag/app/chroma"):
        if embedding is not None:
            self.embedding = embedding
        elif embedding_model is not None:
            self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        else:
            raise ValueError("Either embedding or embedding_model must be provided")

        self.persist_dir = persist_dir
        self.collection_name = "chunked_documents"


    def build_store(self,documents: list[Document]):

        id_chunked_doc = self.add_chunk_id(documents)
        ids = [str(chunk.metadata["chunk_id"]) for chunk in id_chunked_doc]

        Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            ids=ids,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )


    def add_chunk_id(self,documents: list[Document]):
    
        total = 1
        for chunk in documents:
            chunk.metadata.update({
                "chunk_id": total
            })
            total += 1
    
        return documents