import os
from groq import Groq
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()
# Environment
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRADB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRADB_APPLICATION_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# Embedding model (same as ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Astra connection
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    collection_name="astra_vector_langchain",  # replace if different
)

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# Adding Hybrid search for better results obtain
from collections import defaultdict
from typing import List
from langchain_core.documents import Document
from astrapy import DataAPIClient
import os
from langchain_core.documents import Document

def load_all_docs_from_astra():
    client = DataAPIClient(os.getenv("ASTRADB_APPLICATION_TOKEN"))

    db = client.get_database_by_api_endpoint(
        os.getenv("ASTRADB_API_ENDPOINT")
    )

    collection = db.get_collection("astra_vector_langchain")

    docs = []

    cursor = collection.find({}, limit=1000)

    for item in cursor:
        docs.append(
            Document(
                page_content=item.get("content", ""),  # confirm field name
                metadata=item.get("metadata", {})
            )
        )

    return docs



class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, w_dense=0.5, w_sparse=0.5):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.w_dense = w_dense
        self.w_sparse = w_sparse

    def invoke(self, query: str, k: int = 15) -> List[Document]:
        dense_docs = self.dense.get_relevant_documents(query)
        sparse_docs = self.sparse.get_relevant_documents(query)

        scores = defaultdict(float)
        doc_map = {}

        # Dense ranking score
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.metadata.get("scene_number", id(doc))
            scores[doc_id] += self.w_dense * (1 / (rank + 1))
            doc_map[doc_id] = doc

        # Sparse ranking score
        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.metadata.get("scene_number", id(doc))
            scores[doc_id] += self.w_sparse * (1 / (rank + 1))
            doc_map[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_map[doc_id] for doc_id, _ in ranked[:k]]

all_docs = load_all_docs_from_astra()

sparse_retriever = BM25Retriever.from_documents(all_docs)
sparse_retriever.k = 15

dense_retriever = vector_store.as_retriever(search_kwargs={"k": 15})





def retrieve(question, k=8):
    hybrid = HybridRetriever(dense_retriever, sparse_retriever)
    return hybrid.invoke(question, k=k)


def generate_answer(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
You are a Iron man movie analyzer analyzing the movie Iron Man (2008).
At any point of time don't reveal that you are chatgpt or you are backed by openai. If asked that kind of a question reply with - You are intelligent bro don't do this - 
Use ONLY the context below to answer clearly and concisely.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",   # Fast + free tier
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content
