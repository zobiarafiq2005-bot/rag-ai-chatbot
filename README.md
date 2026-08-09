# Enterprise Knowledge Assistant — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB,
and Groq, supporting PDF, DOCX, TXT, and XLSX document ingestion with a
conversational Gradio interface.

## Project Overview

This project implements a multi-format document Q&A chatbot. Users upload
documents, which are chunked, embedded, and stored in a local ChromaDB vector
database. Questions are answered by retrieving the most relevant chunks and
passing them, along with conversation history, to an LLM for a grounded,
natural-language response.

## Objectives

- Ingest and chunk documents across multiple formats
- Generate embeddings and store them in a persistent vector database
- Retrieve relevant context for a given user query
- Apply prompt engineering to ground LLM answers in retrieved context
- Support multi-turn conversational question answering

## System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Document   │────▶│  Document Loader  │────▶│  Text Chunker  │
│ (PDF/DOCX/  │     │ (format routing)  │     │ (Recursive     │
│  TXT/XLSX)  │     └──────────────────┘     │  Splitter)     │
└─────────────┘                              └───────┬────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────┐
                                          │  HuggingFace Embeddings │
                                          │   (all-MiniLM-L6-v2)    │
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────┐
                                          │   ChromaDB Vector Store │
                                          │  (persisted to disk)    │
                                          └────────────┬────────────┘
                                                       │
                                    User Query ────────┤
                                                       ▼
                                          ┌────────────────────────┐
                                          │   Retriever (top-k)     │
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────┐
                                          │  Prompt Construction    │
                                          │ (system + history +     │
                                          │  context + query)       │
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────┐
                                          │   Groq LLM (generation) │
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────┐
                                          │   Answer + Chat History │
                                          │      (Gradio UI)        │
                                          └────────────────────────┘
```

## Flow Diagram

**Ingestion flow:** Upload → Detect format → Load → Split into chunks → Embed
→ Store in ChromaDB → Confirm success in UI.

**Query flow:** User asks question → Retrieve top-k similar chunks → Build
prompt (system instructions + recent history + retrieved context + question)
→ Call LLM → Return answer → Append turn to chat history.

## State Diagram

```
[Idle] --upload file--> [Processing Document] --success--> [Indexed]
[Indexed] --ask question--> [Retrieving Context]
[Retrieving Context] --context found--> [Generating Answer] --> [Idle]
[Retrieving Context] --no context--> [No Answer] --> [Idle]
[Processing Document] --error--> [Ingestion Error] --> [Idle]
[Generating Answer] --LLM error--> [Fallback: Raw Context Shown] --> [Idle]
```

## Methodology

1. Documents are routed to a format-specific LangChain loader based on file
   extension.
2. Loaded documents are split using `RecursiveCharacterTextSplitter`
   (chunk size 1000, overlap 200).
3. Chunks are embedded using a local HuggingFace sentence-transformer model
   (`all-MiniLM-L6-v2`) — no API key required for embeddings.
4. Embeddings are stored in ChromaDB, persisted to disk so the index survives
   restarts.
5. On each question, the top-3 most similar chunks are retrieved and
   combined with a system prompt that instructs the LLM to answer only from
   the provided context.
6. The last 5 conversation turns are included in the prompt for multi-turn
   continuity.
7. Groq's hosted LLM (`llama-3.1-8b-instant`) generates the final answer.

## Technology Stack

| Component            | Technology                          |
|-----------------------|--------------------------------------|
| Orchestration          | LangChain                           |
| Vector Database        | ChromaDB                            |
| Embeddings              | HuggingFace `all-MiniLM-L6-v2`      |
| LLM (generation)        | Groq (`llama-3.1-8b-instant`)       |
| UI                       | Gradio                              |
| Document parsing         | pypdf, unstructured, docx2txt       |

## Installation & Execution (PyCharm)

1. Open this folder as a PyCharm project (**File → Open**).
2. When prompted, let PyCharm create a virtual environment, or manually:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your free Groq API key
   (get one at https://console.groq.com → API Keys):
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
5. Run the app:
   ```bash
   python app.py
   ```
6. Open the local URL shown in the terminal (typically `http://127.0.0.1:7860`).
7. Upload a document, click **Process & Index Document**, then ask a question.

## Experimental Results

- Successfully ingested and indexed PDF documents (e.g. a 3-page PDF split
  into 23 chunks).
- Retrieval correctly surfaces relevant chunks for topical questions.
- Generation via Groq returns grounded answers in under ~2 seconds per query.

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| Gradio 6 removed the `type` parameter and tuple-format chat history | Switched to the dict-based `{"role": ..., "content": ...}` messages format required by current Gradio versions |
| Vector store object was discarded after creation, so nothing could query it | Refactored `VectorStoreManager` to cache a live `Chroma` instance and expose `as_retriever()` |
| Local LLM inference (TinyLlama) was too slow on an 8GB RAM laptop | Switched generation to Groq's cloud API, which offloads compute and returns responses in ~1 second |
| Hugging Face Inference API token permissions caused repeated 401 errors | Avoided the issue entirely by using Groq instead |

## Future Improvements

- Add LangGraph-based multi-step agent workflows (advanced-level task)
- Add source citations in the UI (which document/page an answer came from)
- Support streaming responses token-by-token in the chat UI
- Add automated evaluation of retrieval quality (e.g. Ragas)

## References

- LangChain documentation: https://python.langchain.com
- ChromaDB documentation: https://docs.trychroma.com
- Groq documentation: https://console.groq.com/docs
- Gradio documentation: https://www.gradio.app/docs

## GitHub Repository

_Add your repository link here after pushing this project._