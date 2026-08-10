import json
import os
from dotenv import load_dotenv
from langsmith import traceable
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict

load_dotenv()
JUDGE_MODEL = "gpt-4o-mini"

def _judge(schema):
    return ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
        ).with_structured_output(schema, method="json_schema", strict=True)


def create_dataset_langsmith(dataset_name: str,dataset_description: str,client):
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=dataset_description
    )

    with open("Langsmith_Evaluation_dataset.json","r",encoding="utf-8") as json_file:
        dataset_content = json.load(json_file)

    
    client.create_examples(
        dataset_id = dataset.id,
        examples = dataset_content
    )


@traceable(name="eval_correctness")
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Correctness: Response vs. reference answer."""

    class CorrectnessGrade(TypedDict):
        explanation: Annotated[str, ..., "Step-by-step reasoning for the score."]
        correct: Annotated[bool, ..., "True if the answer is factually correct."]

    CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz.
        You will be given a QUESTION, the GROUND TRUTH ANSWER, and the STUDENT ANSWER.

        Grade criteria:
        (1) Grade ONLY on factual accuracy relative to the ground truth answer.
        (2) The student answer must not contain any statements that conflict with the ground truth.
        (3) It is OK if the student answer has more information than the ground truth,
        as long as it is factually accurate relative to it.

        Correct = True only if ALL criteria are met.
        Explain your reasoning step by step before giving the final grade."""

    correctness_llm = _judge(CorrectnessGrade)

    try:
        user = (
            f"QUESTION: {inputs['question']}\n"
            f"GROUND TRUTH ANSWER: {reference_outputs['answer']}\n"
            f"STUDENT ANSWER: {outputs['answer']}"
        )
        grade = correctness_llm.invoke(
            [{"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
             {"role": "user", "content": user}]
        )
        return grade["correct"]
    except Exception as e:
        return False

@traceable(name="eval_relevance")
def relevance(inputs: dict, outputs: dict) -> bool:
    """Relevance: Response vs. input question (reference-free)."""

    class RelevanceGrade(TypedDict):
        explanation: Annotated[str, ..., "Reasoning for the relevance score."]
        relevant: Annotated[bool, ..., "True if the answer addresses the question."]

    relevance_llm = _judge(RelevanceGrade)

    RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz.
        You will be given a QUESTION and a STUDENT ANSWER.

        Grade criteria:
        (1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION.
        (2) Ensure it helps to answer the QUESTION.

        Relevant = True only if BOTH criteria are met.
        Explain your reasoning step by step before giving the final grade."""
    try:
        user = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = relevance_llm.invoke(
            [{"role": "system", "content": RELEVANCE_INSTRUCTIONS},
             {"role": "user", "content": user}]
        )
        return grade["relevant"]
    except Exception as e:
        return False



