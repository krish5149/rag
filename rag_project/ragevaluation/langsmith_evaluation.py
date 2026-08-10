import json
import os
import asyncio
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.async_client import AsyncClient
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langsmith.evaluation import aevaluate
from app.graphs.query_graph import query_app as rag_pipeline
from app.schemas.langsmithEvalLLMSchema import (
    CorrectnessGrade, RelevanceGrade,
    GroundedGrade, RetrievalRelevanceGrade
)

load_dotenv()
client = AsyncClient()
JUDGE_MODEL = "gpt-4o-mini"
DATASET_DESCRIPTION = "Production RAG evaluation dataset"
DATASET_NAME = "Production RAG dataset"

def judge_llm(schema):

    return ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    ).with_structured_output(schema, method="json_schema", strict=True)


async def create_dataset():

    try:
        dataset = await client.read_dataset(dataset_name=DATASET_NAME)
        return dataset  # already exists, skip creation
    except Exception:
        pass

    dataset = await client.create_dataset(
        dataset_name = DATASET_NAME,
        description = DATASET_DESCRIPTION
    )

    with open("/workspaces/rag/app/ragevaluation/evaluation_dataset.json", "r", encoding="utf-8") as json_files:
        dataset_content = json.load(json_files)

    semaphore = asyncio.Semaphore(5)

    async def _create_one(example):
        async with semaphore:
            await client.create_example(
                dataset_id=dataset.id,
                inputs={"question": example["question"]},
                outputs={"answer": example["ground_truth"]}
            )

    await asyncio.gather(*(_create_one(ex) for ex in dataset_content))


async def run_rag(inputs: dict) -> dict:
    
    result = await rag_pipeline.ainvoke({
        "query": inputs["question"],
        "retrieved_chunks": [],
        "reranked_docs": [],
        "answer": "",
        "relevent_or_not": None,
        "query_rewrite_count": 0
    })

    return {
        "answer": result["answer"],
        "context": "\n\n".join(text for text, score in result["reranked_docs"])
    }


@traceable(name="eval_correctness")
async def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Correctness: Response vs. reference answer."""

    CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz.
        You will be given a QUESTION, the GROUND TRUTH ANSWER, and the STUDENT ANSWER.

        Grade criteria:
        (1) Grade ONLY on factual accuracy relative to the ground truth answer.
        (2) The student answer must not contain any statements that conflict with the ground truth.
        (3) It is OK if the student answer has more information than the ground truth,
        as long as it is factually accurate relative to it.

        Correct = True only if ALL criteria are met.
        Explain your reasoning step by step before giving the final grade."""

    correctness_llm = judge_llm(CorrectnessGrade)

    try:
        user = (
            f"QUESTION: {inputs['question']}\n"
            f"GROUND TRUTH ANSWER: {reference_outputs['answer']}\n"
            f"STUDENT ANSWER: {outputs['answer']}"
        )
        grade = await correctness_llm.ainvoke(
            [{"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
             {"role": "user", "content": user}]
        )
        return grade["correct"]
    except Exception:
        return False


@traceable(name="eval_relevance")
async def relevance(inputs: dict, outputs: dict) -> bool:
    """Relevance: Response vs. input question (reference-free)."""

    RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz.
        You will be given a QUESTION and a STUDENT ANSWER.

        Grade criteria:
        (1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION.
        (2) Ensure it helps to answer the QUESTION.

        Relevant = True only if BOTH criteria are met.
        Explain your reasoning step by step before giving the final grade."""

    relevance_llm = judge_llm(RelevanceGrade)

    try:
        user = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = await relevance_llm.ainvoke(
            [{"role": "system", "content": RELEVANCE_INSTRUCTIONS},
             {"role": "user", "content": user}]
        )
        return grade["relevant"]
    except Exception:
        return False
    

@traceable(name="eval_groundedness")
async def groundedness(inputs: dict, outputs: dict) -> bool:
    """Groundedness: Response vs. retrieved docs (reference-free)."""
    
    GROUNDEDNESS_INSTRUCTIONS = """You are a teacher grading a quiz.
        You will be given FACTS (retrieved context) and a STUDENT ANSWER.

        Grade criteria:
        (1) Ensure the STUDENT ANSWER is grounded in / supported by the FACTS.
        (2) The STUDENT ANSWER must not contain information outside the scope of the FACTS.

        Grounded = True only if BOTH criteria are met (i.e. no hallucination).
        Explain your reasoning step by step before giving the final grade."""

    relevance_llm = judge_llm(GroundedGrade)

    try:
        docs = outputs.get("context", "")
        user = f"FACTS:\n{docs}\n\nSTUDENT ANSWER: {outputs['answer']}"
        grade = await relevance_llm.ainvoke(
            [{"role": "system", "content": GROUNDEDNESS_INSTRUCTIONS},
             {"role": "user", "content": user}]
        )
        return grade["grounded"]
    except Exception as e:
        return False


@traceable(name="eval_retrieval_relevance")
async def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """Retrieval Relevance: Retrieved docs vs. input question (reference-free)."""

    RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz.
        You will be given a QUESTION and a set of retrieved FACTS.

        Grade criteria:
        (1) Ensure the FACTS are relevant to the QUESTION.
        (2) A relevant set of facts contains keywords or semantic meaning related to the QUESTION.

        Relevant = True if the FACTS contain ANY information useful for answering the QUESTION.
        Explain your reasoning step by step before giving the final grade."""

    relevance_llm = judge_llm(RetrievalRelevanceGrade)

    try:
        docs = outputs.get("context", "")
        user = f"QUESTION: {inputs['question']}\n\nFACTS:\n{docs}"
        grade = await relevance_llm.ainvoke(
            [{"role": "system", "content": RETRIEVAL_RELEVANCE_INSTRUCTIONS},
             {"role": "user", "content": user}]
        )
        return grade["relevant"]
    except Exception as e:
        return False


async def main():

    # Step 1: create dataset first (must finish before evaluation starts)
    await create_dataset()

    # Step 2: evaluation runs in parallel, capped at CONCURRENCY_LIMIT
    results = await aevaluate(
        run_rag,
        data=DATASET_NAME,
        evaluators=[correctness, relevance, groundedness, retrieval_relevance],    #type: ignore
        experiment_prefix="rag-eval-v2",
        max_concurrency=2,
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())