from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("write a story on {story} based on Yedu Chepala Katha in telugu:\n")

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.7
)

chain = prompt | llm
response = chain.invoke({"story": "fish"})

print(response.content)
