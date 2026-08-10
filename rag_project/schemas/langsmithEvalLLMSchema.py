from pydantic import BaseModel
from typing import Annotated, TypedDict

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Step-by-step reasoning for the score."]
    correct: Annotated[bool, ..., "True if the answer is factually correct."]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the relevance score."]
    relevant: Annotated[bool, ..., "True if the answer addresses the question."]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]