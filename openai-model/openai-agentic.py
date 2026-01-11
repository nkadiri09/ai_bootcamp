from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template('''
From the below news article, extract a revenue, eps, and forecast in JSON format containing keys 
"year",
"revenue_actual",
"revenue_expected", 
"eps_actual", and 
"eps_expected".
 add only there values 
only json format no preamble.

Guardrails:
include "Year" as a year don not include as fiscal year

{article}
''')

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.7
)

chain = prompt | llm
response = chain.invoke({"article": '''For the most recent quarter ending November 2025 (Fiscal 2026), Oracle reported revenue of $16.1 billion, which narrowly missed the analyst consensus of $16.2 billion. Despite this slight top-line miss, the company posted adjusted earnings of $2.26 per share, significantly outperforming the expected $1.64 EPS. However, as noted, this earnings "beat" was largely inflated by a $2.7 billion pretax gain from the Ampere stake sale; without that one-time boost, earnings would have been much closer to the projected target.

In the prior year’s quarter ending November 2024 (Fiscal 2025), Oracle recorded $14.1 billion in revenue, which met the consensus estimate of $14.1 billion. Its adjusted earnings for that period were $1.47 per share, successfully beating the Wall Street expectation of $1.38 EPS. This period was characterized by the first major signs of the "AI boom" for Oracle, as its cloud infrastructure revenue grew by 52%, matching high investor expectations for its expansion into the GPU-rental market.

Going back to November 2023 (Fiscal 2024), Oracle reported revenue of $12.9 billion, which fell short of the $13.1 billion analysts had anticipated. During this quarter, adjusted earnings came in at $1.34 per share, a modest beat compared to the expected $1.32 EPS. At the time, investors were concerned about the slower-than-expected integration of Cerner, though the company’s infrastructure business was already showing the 52% growth rate that would become its hallmark in the years to follow.'''})

print(response.content)
