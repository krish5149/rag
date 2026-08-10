import os
from dotenv import load_dotenv
from pydantic import SecretStr
from typing_extensions import Annotated, TypedDict
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langsmith import traceable, Client
from working_files.Langsmith_Evaluation_02 import create_dataset_langsmith, correctness, relevance

load_dotenv()

@traceable(name="Load Documents")
def load_documents(url: str) -> list[Document]:
  """Fetch LangChain documentation pages as Documents."""

  loader = WebBaseLoader(url)
  docs = loader.load()
  return docs 

@traceable(name="Split Documents")
def split_documemts(docs: list[Document],chunk_size: int, chunk_overlap: int) -> list[Document]:

  splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
  )

  list_documents = splitter.split_documents(docs)
  return list_documents

def embedding_model() -> HuggingFaceEmbeddings:

  embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
  )
  return embeddings

@traceable(name="Retrieve Documents")
def chroma_db(chunks: list[Document],question: str):

  vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model()
  )

  retriever = vector_store.as_retriever(k=5)
  return retriever.invoke(question)

@traceable(name="Generate Answer")
def generated_llm(context: str,question: str,llm: ChatOpenAI):
  
  template = """
        Answer the Question based on only the following context.

        Context: 
        {context}

        Question:
        {question}
    """
  prompt = ChatPromptTemplate.from_template(template)
  chain = prompt | llm

  return chain.invoke({
    "context" : context,
    "question" : question
  })


@traceable(name="RAG Pipeline")
def run_rag(output: dict):

  question = output["question"]
  retrieved_doc = chroma_db(chunks,question)

  context = "\n\n".join(
    document.page_content for document in retrieved_doc
  )

  answer = generated_llm(context,question,model)
  return {
    "answer" : answer.content
  }



if __name__ == "__main__":

  client = Client()

  # Loading Documents from webpage
  url = "https://python.langchain.com/docs/tutorials/rag/"
  webpage_content = load_documents(url)
  
  # Splitting Documents
  chunks = split_documemts(webpage_content,1000,200)

  model = ChatOpenAI(
      model="openai/gpt-4o-mini",
      api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
      base_url="https://openrouter.ai/api/v1",
      temperature=0,
    )

  # create_dataset_langsmith("rag_dataset","Storing of JSON Values",client)

  results = client.evaluate(
    run_rag,
    data="rag_dataset",
    evaluator=[relevance],  
    experiment_prefix="rag-eval-v2"
) # type: ignore

