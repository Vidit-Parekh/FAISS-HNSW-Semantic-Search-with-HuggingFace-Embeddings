from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np

dimension = 768
k = 5  # nearest neighbors to retrieve

with open("nvidia.txt", "r") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)

chunks = text_splitter.split_text(text)

embeddings = HuggingFaceEmbeddings()

print("Embedding document chunks...")
vectors = embeddings.embed_documents(chunks)
X = np.array(vectors).astype("float32")  # FAISS requires float32

# Build FAISS IndexHNSWFlat
# M = number of connections per node in HNSW graph (16–64 is normal)
M = 32
index = faiss.IndexHNSWFlat(dimension, M)

index.hnsw.efConstruction = 200
index.hnsw.efSearch = 50

print("Adding vectors to HNSW index...")
index.add(X)

query = "chips for A.I."
query_vec = np.array([embeddings.embed_query(query)], dtype="float32")

distances, indices = index.search(query_vec, k)

print("\nTop-k nearest results (HNSW):")
print("Distances:", distances)
print("Indices:", indices)

print(" Retrieved Chunks\n\n ")

for i in range(k):
    print(f"Result #{i+1}:")
    print(chunks[indices[0][i]])

print('\n')
