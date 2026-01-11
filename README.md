# FAISS-HNSW-Semantic-Search-with-HuggingFace-Embeddings

## 📌 Project Overview

This project demonstrates how to build a **semantic search system** using **HuggingFace embeddings**, **LangChain text splitters**, and **FAISS (Facebook AI Similarity Search)** with an **HNSW (Hierarchical Navigable Small World)** index.

The system:

* Loads a real-world news article about **NVIDIA and AI chips**
* Splits the text into manageable chunks
* Converts each chunk into dense vector embeddings
* Stores embeddings in a FAISS HNSW index for efficient similarity search
* Retrieves the most relevant text chunks for a given natural language query

This project is ideal for learning the foundations behind:

* Vector databases
* Semantic search
* Retrieval-Augmented Generation (RAG) pipelines

---

## 🚀 Features

* ✅ Uses **HuggingFace sentence embeddings**
* ✅ Efficient similarity search with **FAISS HNSW index**
* ✅ Chunking using **RecursiveCharacterTextSplitter**
* ✅ Fast Approximate Nearest Neighbor (ANN) retrieval
* ✅ Easily extendable for RAG, chatbots, or document QA

---

## 🗂️ Project Structure

```
.
├── Assignment 8.py        # Main Python script
├── nvidia.txt             # Source document used for semantic search
├── README.md              # Project documentation
```

---

## 🧠 How It Works

### 1️⃣ Load the Document

The project reads a text file (`nvidia.txt`) containing a news article about NVIDIA's dominance in AI chips.

### 2️⃣ Text Chunking

The document is split into fixed-size chunks using:

```python
RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
```

This improves embedding quality and retrieval accuracy.

### 3️⃣ Generate Embeddings

Each chunk is converted into a **768-dimensional dense vector** using HuggingFace embeddings.

### 4️⃣ Build FAISS HNSW Index

* Index type: `IndexHNSWFlat`
* Graph connections (M): 32
* Optimized for fast similarity search

### 5️⃣ Semantic Query Search

A natural language query (e.g., *"chips for A.I."*) is embedded and compared against stored vectors to retrieve the **top-k most relevant chunks**.

---

## 🛠️ Technologies Used

* **Python 3.9+**
* **LangChain**
* **HuggingFace Transformers**
* **FAISS**
* **NumPy**

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install langchain faiss-cpu numpy transformers sentence-transformers
```

> ⚠️ If you have a GPU and CUDA installed, you may use `faiss-gpu` instead of `faiss-cpu`.

---

## ▶️ How to Run

```bash
python main.py
```

---

## 📊 Sample Output

* Displays distances and indices of the nearest neighbors
* Prints the **top-k most relevant text chunks** related to the query

Example:

```
Top-k nearest results (HNSW):
Distances: [[...]]
Indices: [[...]]

Retrieved Chunks:
Result #1:
Nvidia controls about 90 percent of the market for the chips used in A.I. projects...
```

---

## 🔍 Key Parameters Explained

| Parameter        | Description                           |
| ---------------- | ------------------------------------- |
| `dimension`      | Embedding vector size (768)           |
| `k`              | Number of nearest neighbors retrieved |
| `M`              | HNSW graph connections per node       |
| `efConstruction` | Index construction accuracy           |
| `efSearch`       | Search accuracy vs speed trade-off    |

---

## 🧩 Possible Enhancements

* 🔹 Add persistence using FAISS index saving/loading
* 🔹 Integrate with an LLM for **RAG-based Q&A**
* 🔹 Support PDF/HTML documents
* 🔹 Add Streamlit or FastAPI UI
* 🔹 Switch to OpenAI or other embedding models

---

## 🎯 Learning Outcomes

By completing this project, you will understand:

* How semantic search works
* Vector embeddings and similarity metrics
* FAISS indexing strategies (Flat vs HNSW)
* Foundations of Retrieval-Augmented Generation systems

---

## 👨‍💻 Author

**Vidit Parekh**
Master's in  Computer Science
University of Cincinnati

---

## 📜 License

This project is for **educational purposes**. Feel free to fork, modify, and build upon it.

---

⭐ If you found this project helpful, consider giving it a star!
