from langchain_core.documents import Document

employee_kb_chunks = [
    Document(
        page_content="Employees are entitled to 18 days of paid annual leave per calendar year. "
                     "Leave accrues monthly and unused leave can be carried forward up to 5 days "
                     "into the next year, subject to manager approval.",
        metadata={"source": "hr_policy_leave.pdf", "section": "Annual Leave"},
    ),
    Document(
        page_content="Sick leave is capped at 12 days per year and does not require prior approval, "
                     "but a medical certificate is mandatory for absences longer than 2 consecutive days.",
        metadata={"source": "hr_policy_leave.pdf", "section": "Sick Leave"},
    ),
    Document(
        page_content="New employees are subject to a probation period of 90 days from their date of joining. "
                     "During probation, either party may terminate employment with 7 days written notice.",
        metadata={"source": "hr_policy_employment.pdf", "section": "Probation"},
    ),
    Document(
        page_content="Employees working remotely must be available on core hours from 10 AM to 4 PM IST "
                     "and are expected to attend all scheduled team meetings via video call.",
        metadata={"source": "hr_policy_remote.pdf", "section": "Remote Work"},
    ),
    Document(
        page_content="The company reimburses up to INR 5000 per month for home internet and electricity "
                     "for employees on the remote work policy, submitted via the expense portal.",
        metadata={"source": "hr_policy_remote.pdf", "section": "Remote Work Reimbursement"},
    ),
    Document(
        page_content="Performance reviews are conducted twice a year, in April and October. "
                     "Salary revisions, if applicable, are typically processed following the April review cycle.",
        metadata={"source": "hr_policy_performance.pdf", "section": "Performance Reviews"},
    ),
    Document(
        page_content="The company's founding year was 2011, headquartered in Bangalore, "
                     "with additional offices opened in Pune and Hyderabad in 2016 and 2019 respectively.",
        metadata={"source": "company_history.pdf", "section": "Company Overview"},
    ),
]


from openai import OpenAI
response_client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=employee_kb_chunks,
    embedding=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

test_data = [
    # {
    #     "question": "How many days of annual leave do employees get?",
    #     "contexts": [],
    #     "answer": "Employees get 18 days of paid annual leave per year, with up to 5 days carried forward with manager approval.",
    #     "ground_truth": "Employees get 18 days of paid annual leave per year, with up to 5 days carry-forward allowed."
    # },
    # {
    #     "question": "What is the probation period for new employees?",
    #     "contexts": [],
    #     "answer": "New employees have a 90-day probation period, and either party can terminate with 7 days written notice.",
    #     "ground_truth": "New employees have a 90-day probation period, terminable with 7 days written notice by either party."
    # },
    {
        "question": "Is there reimbursement for remote work expenses?",
        "contexts": [],
        "answer": "Yes, employees can get up to INR 5000 per month reimbursed for internet and electricity under the remote work policy.",
        "ground_truth": "Yes, employees are reimbursed up to INR 5000 monthly for internet and electricity under the remote work policy."
    }
]

def generating_llm(question: str,retrieved_contexts: list) -> str:

    context_text = "\n\n".join(
        f"Context {index + 1}:\n{context}"
        for index, context in enumerate(retrieved_contexts)
    )

    completion = response_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided contexts. "
                    "If the answer is not present, say that the available "
                    "information is insufficient."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Retrieved contexts:\n{context_text}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0,
    )

    response = completion.choices[0].message.content or ""
    return response

class RAGEvaluator:

    def __init__(self,retriever,judge_llm):
        self.retriever = retriever
        self.judge_llm = judge_llm

    def run_pipeline(self):
        for tc in test_data:
            docs = self.retriever.invoke(tc['question'])
            context_texts = [d.page_content for d in docs]
            tc['contexts'] = context_texts

eval = RAGEvaluator(vector_retriever,response_client) 
eval.run_pipeline()


# ==================== ContextPrecision ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer_ContextPrecision = ContextPrecision(llm=llm)

# Evaluate

print("=========== Context Precision ===========")

for data in test_data:
    result = scorer_ContextPrecision.score(
        user_input=data["question"],
        reference=data["answer"],
        retrieved_contexts=data["contexts"]
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"Context Precision Score: {result.value}")
print("  ")


# ==================== ContextRecall ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer_ContextRecall = ContextRecall(llm=llm)

# Evaluate

print("=========== Context Recall ===========")

for data in test_data:
    result2 = scorer_ContextRecall.score(
        user_input=data["question"],
        reference=data["answer"],
        retrieved_contexts=data["contexts"]
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"Context Recall Score: {result2.value}")
print("  ")


# ===================================== Response from LLM not retrieved_contexts ========================================


# ==================== Faithfulness ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer_Faithfulness = Faithfulness(llm=llm)

# Evaluate

print("=========== Faithfulness ===========")

for data in test_data:
    result3 = scorer_Faithfulness.score(
        user_input=data["question"],
        response=generating_llm(data["question"],data["contexts"]),
        retrieved_contexts=data["contexts"]
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"Faithfulness Score: {result3.value}")
print("  ")


# ==================== AnswerRelevancy ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

# Create metric
scorer_AnswerRelevancy = AnswerRelevancy(llm=llm, embeddings=embeddings) # type: ignore

# Evaluate

print("=========== AnswerRelevancy ===========")

for data in test_data:
    result3 = scorer_AnswerRelevancy.score(
        user_input=data["question"],
        response=generating_llm(data["question"],data["contexts"])
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"Answer Relevancy Score: {result3.value}")
print("  ")


# ==================== AnswerAccuracy ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer_AnswerAccuracy = AnswerAccuracy(llm=llm)

# Evaluate

print("=========== AnswerAccuracy ===========")

for data in test_data:
    result4 = scorer_AnswerAccuracy.score(
        user_input=data["question"],
        response=generating_llm(data["question"],data["contexts"]),
        reference=data["answer"]
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"Answer Accuracy Score: {result4.value}")
print("  ")

# ==================== NoiseSensitivity ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import NoiseSensitivity

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer_NoiseSensitivity = NoiseSensitivity(llm=llm)

# Evaluate

print("=========== Noise Sensitivity ===========")

for data in test_data:
    result5 = scorer_NoiseSensitivity.score(
        user_input=data["question"],
        response=generating_llm(data["question"],data["contexts"]),
        retrieved_contexts=data["contexts"],
        reference=data["answer"]
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"Noise Sensitivity Score: {result5.value}")
print("  ")


# ==================== ContextRelevance ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer_ContextRelevance = ContextRelevance(llm=llm)

# Evaluate

print("=========== ContextRelevance ===========")

for data in test_data:
    result6 = scorer_ContextRelevance.score(
        user_input=data["question"],
        retrieved_contexts=data["contexts"]
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"ContextRelevance Score: {result6.value}")
print("  ")


# ==================== ResponseGroundedness ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ResponseGroundedness

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer_ResponseGroundedness = ResponseGroundedness(llm=llm)

# Evaluate

print("=========== ResponseGroundedness ===========")

for data in test_data:
    result7 = scorer_ResponseGroundedness.score(
        response=generating_llm(data["question"],data["contexts"]),
        retrieved_contexts=data["contexts"]
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"Response Groundedness Score: {result7.value}")
print("  ")


# ==================== ContextEntityRecall ========================

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextEntityRecall

# Setup LLM
client = AsyncOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer_ContextEntityRecall = ContextEntityRecall(llm=llm)

# Evaluate

print("=========== ContextEntityRecall ===========")

for data in test_data:
    result8 = scorer_ContextEntityRecall.score(
        reference=data["answer"],
        retrieved_contexts=data["contexts"]
    )

    # print(f"Question : {data["question"]}")
    # print(f"Answer : {data["answer"]}")
    # print(f"Context : {data["contexts"]}")
    print(f"Context Entity Recall Score: {result8.value}")
print("  ")