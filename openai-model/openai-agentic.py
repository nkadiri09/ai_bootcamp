from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.7
)

chain = prompt | llm
response = chain.invoke({"output_language": "Spanish", "input": "Good bye.", })

print(response.content)
