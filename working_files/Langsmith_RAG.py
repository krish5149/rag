import os

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv()
documents = [
    Document(
        page_content="LangSmith traces and evaluates LLM applications."
    ),
    Document(
        page_content="RAG retrieves relevant documents before generating an answer."
    ),
    Document(
        page_content="OpenRouter provides access to multiple AI models through one API."
    ),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    """Answer using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
)

answer_chain = prompt | model | StrOutputParser()


@traceable(name="simple-openrouter-rag")
def ask(question: str) -> str:
    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content for document in documents
    )

    return answer_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )


if __name__ == "__main__":
    user_question = input("Question: ")
    answer = ask(user_question)

    print(f"\nAnswer: {answer}")