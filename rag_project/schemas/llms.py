from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import os

class InitiateLLMs:

    def __init__(self,generation_model: str,grade_model: str):

        self.genrate_llm = ChatOpenAI(
            model=generation_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        )

        self.grade_llm = ChatOpenAI(
            model=grade_model,
            api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
