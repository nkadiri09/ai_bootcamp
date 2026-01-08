from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template('''
From the below news article, extract a revenue, eps, and forecast in JSON format containing keys "revenue_actual","revenue_expected", "eps_actual", "eps_expected" and "forecast_actual".

only json format no preamble. for each year

{article}
''')



load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.7
)

chain = prompt | llm
response = chain.invoke({"article": '''TSLA earnings per share for the last quarter are 0.50 USD whereas the estimation was 0.56 USD which accounts for -10.42% surprise. Company revenue for the same period amounts to 28.09 B USD despite the estimated figure of 26.54 B USD. Estimated earnings per share for the next quarter are 0.44 USD, and revenue is expected to reach 24.69 B USD. Also watch annual changes over time to get a bigger picture of TSLA earnings per share and'''})

print(response.content)
