***Iron Man (2008) Intelligence Engine***

Hybrid RAG System with AstraDB + BM25 + Groq LLM

A domain-specific Retrieval-Augmented Generation (RAG) system that answers grounded questions about Iron Man (2008) using a hybrid retrieval pipeline combining dense vector search and sparse keyword search.

**Overview**

This project implements a production-style hybrid RAG architecture that:

Uses AstraDB for persistent vector storage

Uses HuggingFace sentence-transformers for embeddings

Uses BM25 for sparse lexical retrieval

Combines both via hybrid rank fusion

Uses Groq LLM (openai/gpt-oss-20b) for grounded answer generation

Displays retrieved scenes transparently in the UI

Prevents hallucination by restricting answers to retrieved context

**The system is optimized for:**

Multi-hop question answering

Named entity precision

Context grounding

Retrieval transparency

**Architecture**
User Query
   ↓
Hybrid Retriever
   ├── Dense Search (AstraDB Vector Store)
   ├── Sparse Search (BM25 In-Memory Index)
   └── Rank Fusion
   ↓
Top-K Retrieved Scenes
   ↓
LLM (Groq)
   ↓
Grounded Answer
   ↓
Streamlit UI

**Key Features**

*Hybrid Retrieval*

Dense semantic search (768-dim embeddings)

Sparse lexical search (BM25)

Weighted rank fusion

Tunable candidate pool

*Hallucination Prevention*

LLM is constrained to retrieved context only

Returns “Not found in context” when relevant information is absent

*Scene-Level Indexing*

Each movie scene stored as an independent document

Metadata includes scene number and location

Enables precise retrieval and citation

*Retrieval Transparency*

Displays source scenes in UI

Shows response latency

Maintains recent query history

**Tech Stack** 
| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| UI               | Streamlit                               |
| Dense Vector DB  | DataStax AstraDB                        |
| Embeddings       | sentence-transformers/all-mpnet-base-v2 |
| Sparse Retrieval | BM25 (LangChain)                        |
| LLM              | Groq (openai/gpt-oss-120b)              |
| Language         | Python 3.9                              |


📁 Project Structure
ironman-intelligence/
│
├── streamlit_app.py     # Frontend UI
├── rag.py               # Retrieval + Generation logic
├── ingestion.py         # Document ingestion (if applicable)
├── corpus.pkl           # Persisted screenplay corpus (for BM25)
├── requirements.txt
└── README.md


⚙️ Setup Instructions
1*Clone the Repository*

     git clone https://github.com/morty649/ironman-intelligence.git
     cd ironman-intelligence

2. *Install Dependencies*

        pip install -r requirements.txt


If needed:

      pip install astrapy streamlit langchain groq sentence-transformers

3️. *Configure Environment Variables*

Create a .env file:

ASTRADB_API_ENDPOINT=your_endpoint
ASTRADB_APPLICATION_TOKEN=your_token
GROQ_API_KEY=your_groq_key

4. *Run the Application*
  
        streamlit run streamlit_app.py

**Some Example Questions**

Who helped Tony build the first suit and what happened to him?

Why does Tony shut down weapons manufacturing?

Who is Obadiah Stane and how is Iron Monger created?

What happens during Tony’s first flight test?

🔍 Retrieval Strategy
Dense Retrieval

Embedding Model: all-mpnet-base-v2

Vector Dimension: 768

Similarity Search: AstraDB

Sparse Retrieval

BM25 over full screenplay corpus

Handles exact entity matches

Improves recall for rare terms

Fusion Strategy

Weighted inverse-rank scoring

Tunable dense/sparse weights

Adjustable candidate pool size

**Hallucination Control**

The LLM is instructed:

Use ONLY the provided context.

If relevant information is not retrieved, the system returns:

The provided excerpts do not contain this information.

This ensures strict grounding.

**Engineering Challenges Solved**

Dense-only retrieval failing on entity-heavy queries

Semantic vs lexical mismatch

Multi-hop question handling

Ranking errors in hybrid fusion

Corpus reconstruction from persistent vector store

Retrieval debugging and evaluation

**Future Improvements**

Cross-encoder reranking

ReAct agent integration

Streaming LLM responses

Multi-movie support

Automated evaluation metrics (Precision@K, retrieval recall)

React frontend migration

**How This Projectis different from a chatbot**

This is not a basic chatbot.

It demonstrates:

Real-world RAG architecture

Hybrid information retrieval engineering

Vector database integration

Retrieval tuning and debugging

Grounded LLM system design

It reflects production-oriented thinking rather than tutorial-level implementation.
