import chromadb
from chromadb.config import Settings

client = chromadb.Client()

"""
# TO persist data to disk
client = chromadb.Client(
    chromadb.config.Settings(persist_directory="./chroma_db")
)
"""

collection = client.get_or_create_collection(
    name="my_documents"
)

documents = [
    "Python is a popular programming language",
    "ChromaDB is a vector database for embeddings",
    "Metadata helps filter search results"
]

metadatas = [
    {"topic": "python", "level": "beginner"},
    {"topic": "database", "type": "vector"},
    {"topic": "search", "feature": "filtering"}
]

ids = ["doc1", "doc2", "doc3"]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("Records added to ChromaDB collection 'my_documents'.")

# query Data from Database.

results = collection.query(
    query_texts=["co-signed search"],
    n_results=2,
    where={"topic": "database"}
)

# for applying multi condition
"""
where={"$and": [{"topic": "python"}, {"level": "beginner"}]}
"""

print(results)
print(results["documents"])
print(results["metadatas"])
