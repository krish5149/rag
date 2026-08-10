from pydantic import BaseModel, Field

class GradeRelevance(BaseModel):

    is_relevant: bool = Field(
        description = "True if the content is relevent to the query"
    )

class RewriteQuery(BaseModel):

    query: str = Field(
        description = "Improved query optimized for retrieva"
    )

class GenerateAnswer(BaseModel):

    answer: str = Field(
        description = "Answer to the user's question based on the retrieved context"
    )

class IntentRouter(BaseModel):

    intent: str = Field(
        description = "Classify to the user's question into one of them: greeting, no"
    )