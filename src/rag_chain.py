import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from groq import Groq

# Groq runs on dedicated inference hardware — typically sub-second responses,
# no local RAM/CPU cost, and a generous free tier. Get a free API key at
# https://console.groq.com -> API Keys, then put it in your .env as GROQ_API_KEY.
DEFAULT_MODEL = "llama-3.1-8b-instant"


class RAGConversationalEngine:
    """Retrieves relevant document context from ChromaDB and generates a
    grounded, natural-language answer using Groq's cloud inference API.

    Takes the VectorStoreManager itself (not a raw Chroma instance), since
    the engine is built at app startup before any document is uploaded —
    the retriever is built fresh on every question instead.
    """

    def __init__(self, vector_manager, model_name: str = DEFAULT_MODEL):
        self.vector_manager = vector_manager
        self.chat_history: List[Dict[str, str]] = []
        self.model_name = model_name

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[WARN] GROQ_API_KEY not found in environment — "
                  "generation will fail until it's set in .env")
        self.client = Groq(api_key=api_key)

    def _build_messages(self, query: str, context_text: str) -> List[Dict[str, str]]:
        """Builds the chat messages sent to the LLM: a system prompt grounding
        the model in the retrieved context, plus recent conversation turns so
        follow-up questions ('what about the other one?') still work."""
        system_prompt = (
            "You are an enterprise knowledge assistant. Answer the user's question "
            "using ONLY the context below, which was retrieved from their uploaded "
            "documents. If the answer isn't contained in the context, say you don't "
            "have enough information in the indexed documents to answer — do not "
            "make anything up. Be concise and clear.\n\n"
            f"CONTEXT:\n{context_text}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        for turn in self.chat_history[-5:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": query})
        return messages

    def ask_question(self, query: str) -> Dict[str, Any]:
        """Processes a user query: retrieves matching context, then generates
        an answer grounded in that context via Groq."""
        print(f"\n[THINKING] Searching relevant context for: '{query}'...")

        try:
            retriever = self.vector_manager.as_retriever(search_kwargs={"k": 3})
        except ValueError:
            return {
                "answer": "⚠️ No documents indexed yet — please upload and process a document first.",
                "source_documents": []
            }

        retrieved_docs: List[Document] = retriever.invoke(query)

        if not retrieved_docs:
            answer = "I couldn't find anything relevant in the uploaded documents to answer that."
        else:
            context_text = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
            messages = self._build_messages(query, context_text)

            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.3,
                )
                answer = completion.choices[0].message.content
            except Exception as e:
                print(f"[LLM ERROR]: {e}")
                answer = (
                    "⚠️ The language model could not be reached, so here is the "
                    f"raw retrieved context instead:\n\n{context_text}\n\n"
                    f"(Error: {e})"
                )

        self.chat_history.append({"user": query, "assistant": answer})

        return {
            "answer": answer,
            "source_documents": retrieved_docs
        }