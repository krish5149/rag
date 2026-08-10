import os
import json
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy,
    AnswerAccuracy, NoiseSensitivity, ContextRelevance, 
    ResponseGroundedness, ContextEntityRecall
)
from app.graphs.query_graph import query_app as rag_pipeline
load_dotenv()

client = AsyncOpenAI(
    api_key = os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

llm = llm_factory("gpt-4o-mini",client=client)
embedding = embedding_factory("openai", model="text-embedding-3-small",client=client)

scorer_context_precision     = ContextPrecision(llm=llm)
scorer_context_recall        = ContextRecall(llm=llm)
scorer_faithfulness          = Faithfulness(llm=llm)
scorer_answer_relevancy      = AnswerRelevancy(llm=llm, embeddings=embedding) #type: ignore
scorer_answer_accuracy       = AnswerAccuracy(llm=llm)
scorer_noise_sensitivity     = NoiseSensitivity(llm=llm)
scorer_context_relevance     = ContextRelevance(llm=llm)
scorer_response_groundedness = ResponseGroundedness(llm=llm)
scorer_context_entity_recall = ContextEntityRecall(llm=llm)

async def evaluate_item(data: dict) -> dict:

    question = data["question"]
    ground_truth = data["ground_truth"]

    result = await rag_pipeline.ainvoke({
        "query": question,
        "retrieved_chunks": [],
        "reranked_docs": [],
        "answer": "",
        "relevent_or_not": None,
        "query_rewrite_count": 0
    })

    context = ["\n\n".join(
        text
        for text, score in result["reranked_docs"]
    )]
    response = result["answer"]

    # Now you have all 4 pieces RAGAS needs:
    # question (user_input), contexts (retrieved_contexts),
    # response (answer from llm), reference (ground_truth)

    contextprecision = await scorer_context_precision.ascore(
        user_input=question,
        reference=ground_truth,
        retrieved_contexts=context
    )

    context_recall = await scorer_context_recall.ascore(
        user_input=question,
        reference=ground_truth,
        retrieved_contexts=context
    )

    faithfulness = await scorer_faithfulness.ascore(
        user_input=question,
        response=response,
        retrieved_contexts=context
    )

    answer_accuracy = await scorer_answer_accuracy.ascore(
        user_input=question,
        response=response,
        reference=ground_truth
    )

    answer_relevancy = await scorer_answer_relevancy.ascore(
        user_input=question,
        response=response
    )

    noise_sensitivity = await scorer_noise_sensitivity.ascore(
        user_input=question,
        response=response,
        retrieved_contexts=context,
        reference=ground_truth
    )

    context_relevance = await scorer_context_relevance.ascore(
        user_input=question,
        retrieved_contexts=context
    )

    response_groundedness = await scorer_response_groundedness.ascore(
        response=response,
        retrieved_contexts=context
    )

    context_entity_recall = await scorer_context_entity_recall.ascore(
        reference=ground_truth,
        retrieved_contexts=context
    )

    return {
        "Query" : question,
        "Response" : response,
        "Ground Truth" : ground_truth,
        "Context" : context,
        "ContextPrecision" : contextprecision.value,
        "Context Recall" : context_recall.value,
        "FaithFulness" : faithfulness.value,
        "Answer Accuracy" :answer_accuracy.value,
        "Answer Relevancy" : answer_relevancy.value,
        "Noise Sensitivity" : noise_sensitivity.value,
        "Context Relevance" : context_relevance.value,
        "Response Groundedness" : response_groundedness.value,
        "Context Entity Recall" : context_entity_recall.value
    }



async def run_evaluation(test_data: list[dict]):

    semaphore = asyncio.Semaphore(2)
    async def guarded(d):
        async with semaphore:
            return await evaluate_item(d)

    result = await asyncio.gather(*(guarded(d) for d in test_data))
    return pd.DataFrame(result)
    

    
test_data = []

with open("app/ragevaluation/evaluation_dataset.json","r") as f:
    test_data = json.load(f)

df = asyncio.run(run_evaluation(test_data))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"/workspaces/rag/app/ragevaluation/ragas_report/ragas_results_{timestamp}.csv"
df.to_csv(output_path, index=False)
print(f"Saved results to {output_path}")
print(df.mean(numeric_only=True))