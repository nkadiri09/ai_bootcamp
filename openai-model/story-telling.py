from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(''' write a short story on {animal} similar to yedu chepala kadha in telugu write in {language} in 300 words ''')

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=1.5, top_p=0.95
)

chain = prompt | llm
response = chain.invoke({"animal": "tiger", "language": "english"})

print(response.content)
