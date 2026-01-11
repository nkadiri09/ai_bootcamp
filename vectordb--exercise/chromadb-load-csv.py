import chromadb
import pandas as pd

client = chromadb.Client()

"""
# TO persist data to disk
client = chromadb.Client(
    chromadb.config.Settings(persist_directory="./chroma_db")
)
"""

collection = client.get_or_create_collection(
    name="medical_notes"
)

df = pd.read_csv("healthcare_patient_records.csv")

# list of columns
# ['patient_id', 'symptoms', 'diagnosis', 'treatment', 'doctor_notes']

# Extract each column as a separate list
patient_id_list = df['patient_id'].tolist()
symptoms_list = df['symptoms'].tolist()
diagnosis_list = df['diagnosis'].tolist()
treatment_list = df['treatment'].tolist()
doctor_notes_list = df['doctor_notes'].tolist()



# Iterate through each row and extract all data
documents = []
metadatas = []
ids = []

for index, row in df.iterrows():
    # Extract each column value
    patient_id = row['patient_id']
    symptoms = row['symptoms']
    diagnosis = row['diagnosis']
    treatment = row['treatment']
    doctor_notes = row['doctor_notes']

    # Prepare data for ChromaDB
    documents.append(doctor_notes)
    metadatas.append({
        'patient_id': patient_id,
        'symptoms': symptoms,
        'diagnosis': diagnosis,
        'treatment': treatment,
        'doctor_notes': doctor_notes
    })
    ids.append(patient_id)

# Add all documents to ChromaDB collection
print(f"\nAdding {len(documents)} documents to ChromaDB collection...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)



print(f"✓ Successfully added {len(documents)} patient records to ChromaDB!")
print(f"✓ Collection '{collection.name}' now contains {collection.count()} documents")

client = chromadb.Client()
collection = client.get_or_create_collection(name="medical_notes")
results = collection.query(
    query_texts=["flu"],
    n_results=2
)
print(results)

print(results["documents"][0])
print(results["metadatas"][0])
