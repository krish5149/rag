from typing import Annotated, TypedDict, Literal, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.documents import Document
from app.schemas.llmschemas import GradeRelevance, RewriteQuery, GenerateAnswer, IntentRouter
from app.graphs.storing_graph import  reranker, llm, store, shared_embedding
from app.ragservices.hybridsearch import HybridSearch
from langchain_core.messages import HumanMessage, AIMessage,BaseMessage
from langgraph.checkpoint.memory import MemorySaver

class RAGStateManagment(TypedDict):
    query: str
    retrieved_chunks: list[Document]
    reranked_docs: list[str]
    answer: str
    relevent_or_not: Optional[bool]
    query_rewrite_count: int
    messages: Annotated[list[str],add_messages] 

PERSIST_DIR = "/workspaces/rag/app/chroma"
retriever = HybridSearch(embedding=shared_embedding, persist_dir=PERSIST_DIR)

def intent_checker(state: RAGStateManagment):

    user_input = state["query"]

    INTENT_CHECK_INSTRUCTIONS = ChatPromptTemplate.from_template(
        """Classify user input "{user_input}" if it is greeting then return as 
        below single word as "greeting" else return "no" """
    )

    structured_llm = llm.genrate_llm.with_structured_output(IntentRouter,method="json_schema", strict=True)

    routing_llm = INTENT_CHECK_INSTRUCTIONS | structured_llm
    llm_result = routing_llm.invoke({"user_input": user_input})

    return llm_result.intent #type: ignore

def greet_user(state: RAGStateManagment):

    hour = datetime.now().hour
    if 5 <= hour < 12:
        greeting = "Good morning"
    elif 12 <= hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    answer = (
        f"{greeting}! I'm here to help with hospital procedures, medication information, "
        f"and patient care questions. What would you like to know?"
    )
    return {"answer": answer}


@traceable(name="Retrieve Documents")
def retrieving(state: RAGStateManagment) -> dict:

    retrieved_docs = retriever.hybrid_search(state["query"])
    return {
        "retrieved_chunks": retrieved_docs
    }

@traceable(name="Reranking Context")
def reranked(state: RAGStateManagment) -> dict:

    reranked_docs = reranker.rerank(state["retrieved_chunks"], state["query"])
    reranked_docs = [(text, float(score)) for text, score in reranked_docs]
    return {
        "reranked_docs": reranked_docs
    }

@traceable(name="Grading Context")
def GradeDocument(state: RAGStateManagment):

    prompt = ChatPromptTemplate.from_template("""
        You are a document relevance grader.

        User Query:
        {query}

        Retrieved Document:
        {document}

        Determine if the document contains information
        useful for answering the query.
    """)

    structured_llm = llm.grade_llm.with_structured_output(
        GradeRelevance
    )

    grade = prompt | structured_llm
    result = grade.invoke({
        "query" : state["query"],
        "document" : state["reranked_docs"]
    })

    return {
        "relevent_or_not" : result.is_relevant #type: ignore
    }

@traceable(name="Generate Answer")
def generate_answer(state: RAGStateManagment):

    prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant.

        Use only the provided context to answer the user's question.

        Guidelines:
        - Answer using information from the context.
        - If the answer cannot be found in the context, say:
        "I could not find sufficient information in the retrieved documents."
        - Do not make up information.
        - Be concise and accurate.
        - Use multiple context passages if relevant.

        User Question:
        {query}

        Context:
        {context}
    """)

    context = "\n\n".join(
        text
        for text, score in state["reranked_docs"]
    )

    structured_llm = llm.genrate_llm.with_structured_output(
        GenerateAnswer
    )

    llm_chain = prompt | structured_llm

    answer = llm_chain.invoke({
        "query" : state["query"],
        "context" : context
    })

    return {
        "answer" : answer.answer    #type: ignore
    }

@traceable(name="Rewriting Query")
def rewrite_query(state: RAGStateManagment):

    prompt = ChatPromptTemplate.from_template("""

        You are an expert query rewriter for a Retrieval-Augmented Generation (RAG) system.

        Your task is to rewrite the user's question so that it is easier for a search engine and vector database to retrieve relevant documents.

        Guidelines:
        - Preserve the original meaning.
        - Make the query more specific and descriptive.
        - Expand ambiguous terms when appropriate.
        - Include important keywords that improve retrieval.
        - Do NOT answer the question.
        - Return only the rewritten query.

        Original Query:
        {query}
    """)

    structured_llm = llm.genrate_llm.with_structured_output(
        RewriteQuery
    )

    rewrite_chain = prompt | structured_llm

    result = rewrite_chain.invoke({
        "query" : state["query"]
    })

    return {
        "query" : result.query,      #type: ignore
        "query_rewrite_count" : state["query_rewrite_count"] + 1
    }

def query_router(state: RAGStateManagment):

    if state["relevent_or_not"]:
        return "generate"
    else :
        if(state["query_rewrite_count"] < 1):
            return "rewrite"
        else:
            return "fallback"

def Fallback(state: RAGStateManagment):

    return {
        "answer":
        "I could not find relevant information in the retrieved documents after multiple retrieval attempts."
    }



graph = StateGraph(RAGStateManagment)
graph.add_node("Retrieving",retrieving)
graph.add_node("Reranking", reranked)
graph.add_node("Grading", GradeDocument)
graph.add_node("Rewriting Query", rewrite_query)
graph.add_node("Generating", generate_answer)
graph.add_node("Fallback", Fallback)
graph.add_node("Greeting", greet_user)


graph.add_conditional_edges(
    START,
    intent_checker,
    {
        "greeting" : "Greeting",
        "no" : "Retrieving"
    }
)
graph.add_edge("Greeting",END)
graph.add_edge("Retrieving", "Reranking")
graph.add_edge("Reranking", "Grading")
graph.add_conditional_edges(
    "Grading",
    query_router,
    {
        "generate" : "Generating",
        "rewrite" : "Rewriting Query",
        "fallback" : "Fallback"
    }
)
graph.add_edge("Fallback",END)
graph.add_edge("Rewriting Query", "Retrieving")
graph.add_edge("Generating", END)

# config = 
memory = MemorySaver()
query_app = graph.compile(checkpointer=memory)

# Print an ASCII diagram of the graph structure
# print(query_app.get_graph().draw_ascii())