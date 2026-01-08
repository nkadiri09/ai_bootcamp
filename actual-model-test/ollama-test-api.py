from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=1.8, num_predict=15
    # other params...
)

print(llm.invoke("provide a recipie for apple pie"))