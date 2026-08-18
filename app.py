import os
import json
from dotenv import load_dotenv

load_dotenv()  # Reads .env into the environment (HF, Groq keys etc.)

import gradio as gr
from src.document_loader import MultiFormatDocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGConversationalEngine

# Initialize core pipeline components once at startup
vector_manager = VectorStoreManager()
rag_engine = RAGConversationalEngine(vector_manager=vector_manager)

CHAT_LOG_PATH = "conversation_history.jsonl"
MAX_RESTORED_TURNS = 20  # cap how many past exchanges get reloaded into the UI


def load_chat_history():
    """Reads the persistent conversation log and rebuilds the chat display
    from it, so the visible chat survives a page refresh instead of always
    starting blank. Note: this log is a single shared file (not tagged per
    browser session), so on a page reload everyone sees the same restored
    history — acceptable for local single-user testing, but worth knowing
    if this app were ever used by multiple people at once."""
    if not os.path.exists(CHAT_LOG_PATH):
        return []

    messages = []
    try:
        with open(CHAT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-MAX_RESTORED_TURNS:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages.append({"role": "user", "content": record.get("query", "")})
            messages.append({"role": "assistant", "content": record.get("answer", "")})
    except Exception as e:
        print(f"[HISTORY RELOAD WARNING] Could not load past chat history: {e}")
        return []

    return messages


def process_file(file):
    """Handles document upload: load, chunk, and index into ChromaDB."""
    if file is None:
        return "⚠️ Please upload a valid document (.pdf, .docx, .txt, .xlsx)."

    try:
        file_path = file.name if hasattr(file, 'name') else str(file)

        loader = MultiFormatDocumentLoader(file_path=file_path)
        chunks = loader.load_and_split()

        vector_manager.add_documents(chunks)

        return f"✅ '{os.path.basename(file_path)}' ({len(chunks)} chunks) successfully indexed into Vector DB!"

    except Exception as e:
        print(f"[INGESTION ERROR]: {str(e)}")
        return f"❌ Ingestion Error: {str(e)}"


def respond(user_message, chat_history):
    """Handles a chat turn: retrieve + generate an answer, update history.

    chat_history is Gradio's own per-session state (a list of
    {"role": ..., "content": ...} dicts). It is passed into
    rag_engine.ask_question() BEFORE the current turn is appended, so the
    engine sees the conversation-so-far and can resolve follow-up questions
    — this is what actually gives the chatbot conversational memory, tied
    correctly to each browser session rather than to a single shared engine
    instance.
    """
    if not user_message or not user_message.strip():
        return "", chat_history

    if chat_history is None:
        chat_history = []

    try:
        result = rag_engine.ask_question(user_message, history=chat_history)
        bot_message = result["answer"]
    except Exception as e:
        print(f"[CHAT ERROR]: {str(e)}")
        bot_message = f"❌ Response Error: {str(e)}"

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_message})

    return "", chat_history


# --- UI ---
custom_css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.app-header {
    background: linear-gradient(135deg, #1e2640 0%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    text-align: left;
}
.app-title {
    color: #f8fafc !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
}
.app-subtitle {
    color: #94a3b8 !important;
    font-size: 13.5px !important;
    margin: 0 !important;
}
.card-panel {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
.custom-btn {
    background: #2563eb !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease-in-out;
}
.custom-btn:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4);
}
"""

with gr.Blocks(title="RAG Enterprise AI") as demo:
    gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">⚡ Enterprise Knowledge Assistant</h1>
            <p class="app-subtitle">Multi-Format RAG Pipeline | Context-Aware Vector Retrieval Engine</p>
        </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, elem_classes=["card-panel"]):
            gr.Markdown("#### 📥 Document Ingestion Hub")

            file_input = gr.File(
                label="Supported Formats: PDF, DOCX, TXT, XLSX",
                file_types=[".pdf", ".docx", ".txt", ".xlsx"],
                type="filepath"
            )

            upload_btn = gr.Button("🚀 Process & Index Document", elem_classes=["custom-btn"])

            status_output = gr.Textbox(
                label="System Logs / Status",
                interactive=False,
                placeholder="Awaiting file selection...",
                lines=2
            )

            upload_btn.click(fn=process_file, inputs=[file_input], outputs=[status_output])

        with gr.Column(scale=6, elem_classes=["card-panel"]):
            gr.Markdown("#### 💬 Conversational Workspace")

            chatbot = gr.Chatbot(
                height=420,
                label="Assistant Workspace",
                show_label=False,
                avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712035.png")
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question based on your indexed document context...",
                    label="Query",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear = gr.ClearButton([msg, chatbot], value="🗑️ Clear Chat Console")
                load_history_btn = gr.Button("📜 Load Previous Conversation")

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot])
            load_history_btn.click(fn=load_chat_history, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), css=custom_css)