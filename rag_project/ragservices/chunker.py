from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Chunker:

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunked_doc = splitter.split_documents(documents)

        total_chunk_count = 1
        for chunk in chunked_doc:
            chunk.metadata.update({
                "chunk_id": total_chunk_count
            })
            total_chunk_count += 1


        return chunked_doc

