from typing import Optional
from app.ragservices.embed_store import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class HybridSearch:

    def __init__(self, embedding: Optional[HuggingFaceEmbeddings] = None, embedding_model: str | None = None, persist_dir: str = "/workspaces/rag/app/chroma"):

        if embedding is not None:
            self.embedding = embedding
        elif embedding_model is not None:
            self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        else:
            raise ValueError("Either embedding or embedding_model must be provided")

        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.bm25_retriever = None
        self._build_vectorstore()
        self._build_bm25()

    def _build_vectorstore(self):
        self.vectorstore = Chroma(
            collection_name="chunked_documents",
            persist_directory=self.persist_dir,
            embedding_function=self.embedding,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def _build_bm25(self):
        data = self.vectorstore.get() #type: ignore
        print("Total Chroma Docs:",len(data["documents"]))

        docs_raw = data.get("documents") or []

        if not docs_raw:
            self.bm25_retriever = None
            return

        documents = [Document(page_content=content) for content in docs_raw]
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 10

    def vector_retriever(self):
        if self.vectorstore is None:
            self._build_vectorstore()

        return self.vectorstore.as_retriever(  #type: ignore
            search_type="similarity",
            search_kwargs={
                "k": 10,
                # "score_threshold": 0.7
            }
        )

    def hybrid_search(self, query):

        vector_docs = self.vector_retriever().invoke(query)

        print("Retrieved Docs Count:", len(vector_docs))
        if vector_docs:
            print(vector_docs[0].page_content[:500])

        if not vector_docs:
            # Vector search found nothing above threshold — treat as out-of-domain,
            # don't let BM25 backfill irrelevant keyword matches.
            return []

        if not self.bm25_retriever:
            return vector_docs

        ensemble = EnsembleRetriever(
            retrievers=[self.vector_retriever(), self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble.invoke(query)