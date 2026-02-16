import os
from groq import Groq
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
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


def retrieve(question, k=8):
    return vector_store.similarity_search(question, k=k)


def generate_answer(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
You are a Iron man movie analyzer analyzing the movie Iron Man (2008).
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
