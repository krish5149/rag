from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

class DocumentLoader:

    def __init__(self,ext: str,filepath: str):
        self.ext = ext
        self.filepath = filepath

    def choosing_loader(self):

        if self.ext.lower() == "pdf":
            loader = PyPDFLoader(self.filepath)

        elif self.ext.lower() == "docx":
            loader = Docx2txtLoader(self.filepath)

        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")

        return loader.load()


