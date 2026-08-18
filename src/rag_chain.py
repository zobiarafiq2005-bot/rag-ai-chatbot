import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from groq import Groq

# Groq runs on dedicated inference hardware — typically sub-second responses,
# no local RAM/CPU cost, and a generous free tier. Get a free API key at
# https://console.groq.com -> API Keys, then put it in your .env as GROQ_API_KEY.
# NOTE: llama-3.1-8b-instant was deprecated and fully shut down by Groq on
# Aug 16, 2026. openai/gpt-oss-20b is Groq's recommended replacement — same
# "fast, small" tier, comparable speed and cost.
DEFAULT_MODEL = "openai/gpt-oss-20b"


class RAGConversationalEngine:
    """Retrieves relevant document context from ChromaDB and generates a
    grounded, natural-language answer using Groq's cloud inference API.

    Conversation history is passed in explicitly per call (as the `history`
    argument to ask_question), rather than stored only as internal engine
    state. This matters because the engine object is created once when the
    app starts and is shared across all requests — if it kept its own
    private chat_history as the sole source of memory, that state would not
    be tied to a specific user's session, and concurrent users would bleed
    into each other's conversations. By accepting the caller's session
    history explicitly, the UI layer (app.py, using Gradio's own per-session
    chat state) stays the single source of truth for "whose conversation is
    this", and this engine just uses whatever history it's given.
    """

    def __init__(self, vector_manager, model_name: str = DEFAULT_MODEL, log_path: str = "conversation_history.jsonl"):
        self.vector_manager = vector_manager
        self.model_name = model_name
        self.log_path = log_path

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[WARN] GROQ_API_KEY not found in environment — "
                  "generation will fail until it's set in .env")
        self.client = Groq(api_key=api_key)

    def _log_turn(self, query: str, answer: str, history_received: int) -> None:
        """Appends this Q&A turn to a persistent log file on disk, separate
        from Gradio's in-memory session state and from the live terminal
        output. This is what lets conversation history be inspected after
        the fact — even after the app is closed or the browser session
        ends — rather than existing only transiently while the app runs."""
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "answer": answer,
            "history_messages_received": history_received,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[LOG WARNING] Could not write to conversation log: {e}")

    def _build_messages(self, query: str, context_text: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Builds the chat messages sent to the LLM: a system prompt grounding
        the model in the retrieved context, plus recent conversation turns
        (passed in by the caller) so follow-up questions
        ('what about the other one?') resolve correctly.

        `history` is expected in the standard messages format:
        [{"role": "user"/"assistant", "content": "..."}, ...] — i.e. exactly
        what Gradio's Chatbot component already stores, so app.py can pass
        its session state straight through with no conversion step.
        """
        system_prompt = (
            "You are an enterprise knowledge assistant. Answer the user's question "
            "using ONLY the context below, which was retrieved from their uploaded "
            "documents. If the answer isn't contained in the context, say you don't "
            "have enough information in the indexed documents to answer — do not "
            "make anything up. Be concise and clear.\n\n"
            f"CONTEXT:\n{context_text}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        history = history or []
        # Keep the last 5 exchanges (10 messages: 5 user + 5 assistant) for context
        for turn in history[-10:]:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})
        return messages

    def ask_question(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Processes a user query: retrieves matching context, then generates
        an answer grounded in that context via Groq.

        `history` — the calling session's conversation so far, in
        [{"role": ..., "content": ...}] format. Pass the UI's own chat state
        here on every call; this method does not store or mutate any memory
        of its own, so it stays safe to share across concurrent sessions.
        """
        print(f"\n[THINKING] Searching relevant context for: '{query}'...")
        print(f"[MEMORY] Received {len(history or [])} prior message(s) in conversation history.")
        if history:
            for i, turn in enumerate(history[-4:]):  # show last few for readability
                role = turn.get("role", "?")
                content = (turn.get("content") or "")[:80]
                print(f"[MEMORY]   [{i}] {role}: {content}{'...' if len(turn.get('content') or '') > 80 else ''}")

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
            messages = self._build_messages(query, context_text, history=history)

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

        self._log_turn(query, answer, history_received=len(history or []))

        return {
            "answer": answer,
            "source_documents": retrieved_docs
        }