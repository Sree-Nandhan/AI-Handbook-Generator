"""
Gradio UI — full-width layout with sidebar chat history.
1. User sees PDF upload area
2. After upload + indexing, chat area appears with sidebar
3. Users can start new chats and revisit old ones
"""

import json
import os
import time
import gradio as gr
from app.rag_engine import RAGEngine
from app.handbook_generator import HandbookGenerator
from app.handlers import handle_upload, handle_chat
from app.config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

CHAT_HISTORY_DIR = os.path.join(OUTPUT_DIR, "chat_history")


def _save_chat(chat_id: str, messages: list, title: str = ""):
    """Save a chat session to disk."""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    data = {
        "id": chat_id,
        "title": title or _derive_title(messages),
        "messages": messages,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_chat(chat_id: str) -> dict | None:
    """Load a chat session from disk."""
    path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _list_chats() -> list[dict]:
    """List all saved chats, newest first."""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    chats = []
    for fname in os.listdir(CHAT_HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(CHAT_HISTORY_DIR, fname)) as f:
                data = json.load(f)
            chats.append({
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at", ""),
            })
        except Exception:
            continue
    return sorted(chats, key=lambda c: c["updated_at"], reverse=True)


def _derive_title(messages: list) -> str:
    """Derive a short title from the first user message."""
    for m in messages:
        if m.get("role") == "user":
            text = m["content"][:50].strip()
            if len(m["content"]) > 50:
                text += "..."
            return text
    return "New Chat"


def _relative_time(timestamp_str: str) -> str:
    """Convert '2026-04-23 23:10:00' to 'Just now', '5m ago', '2h ago', etc."""
    try:
        from datetime import datetime
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        diff = now - ts
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return "Just now"
        elif seconds < 3600:
            return f"{seconds // 60}m ago"
        elif seconds < 86400:
            return f"{seconds // 3600}h ago"
        elif seconds < 604800:
            return f"{seconds // 86400}d ago"
        else:
            return ts.strftime("%b %d")
    except Exception:
        return timestamp_str


def _chat_choices(chats: list) -> list[str]:
    """Build choices for the chat radio selector."""
    choices = []
    for c in chats[:15]:
        title = c["title"][:40]
        rel_time = _relative_time(c.get("updated_at", ""))
        choices.append(f"{title} · {rel_time}")
    return choices


def _chat_id_map(chats: list) -> dict[str, str]:
    """Map display labels to chat IDs."""
    mapping = {}
    for c in chats[:15]:
        title = c["title"][:40]
        rel_time = _relative_time(c.get("updated_at", ""))
        label = f"{title} · {rel_time}"
        mapping[label] = c["id"]
    return mapping


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#f4f6fb",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 1px 4px rgba(0,0,0,0.06)",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4338ca, #6d28d9)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    button_primary_shadow="0 4px 14px rgba(79,70,229,0.25)",
    button_primary_shadow_hover="0 6px 20px rgba(79,70,229,0.35)",
    input_background_fill="white",
    input_border_color="#e2e8f0",
    input_border_color_focus="#6366f1",
    input_border_width="1.5px",
    input_shadow_focus="0 0 0 3px rgba(99,102,241,0.1)",
    border_color_accent="transparent",
    border_color_accent_subdued="transparent",
)

CSS = """
/* ── Full-width layout ── */
.gradio-container { max-width: 100% !important; padding: 0 !important; }
#main-row { min-height: 100vh; gap: 0 !important; }

/* Kill all purple accent borders and outlines */
* {
    --border-color-accent: transparent !important;
    --border-color-accent-subdued: transparent !important;
}
.upload-container, [data-testid="file"], .file-preview {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
/* Remove purple highlight around file upload area */
div[class*="file"] {
    border-color: #e2e8f0 !important;
}
div[class*="file"]:focus-within {
    border-color: #e2e8f0 !important;
    box-shadow: none !important;
}

/* ── Sidebar ── */
#sidebar {
    background: #1e1b4b;
    min-width: 260px;
    max-width: 260px;
    padding: 0;
    border-radius: 0 !important;
    min-height: 100vh;
}
#sidebar-inner {
    padding: 16px 12px;
}
#sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 4px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 12px;
}
#sidebar-header h3 {
    color: white;
    font-size: 1rem;
    font-weight: 700;
    margin: 0;
}
#new-chat-btn {
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    min-height: 34px !important;
    padding: 4px 12px !important;
}
#new-chat-btn:hover {
    background: rgba(255,255,255,0.2) !important;
}

/* Chat list items */
.chat-item {
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 4px;
    transition: background 0.15s;
}
.chat-item:hover { background: rgba(255,255,255,0.08); }
.chat-item.active { background: rgba(99,102,241,0.3); }
.chat-title {
    color: #e2e8f0;
    font-size: 0.85rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.chat-time {
    color: #94a3b8;
    font-size: 0.7rem;
    margin-top: 2px;
}
.chat-empty {
    color: #64748b;
    font-size: 0.85rem;
    text-align: center;
    padding: 20px 0;
}

/* ── Main content area ── */
#main-content {
    flex: 1;
    padding: 0 !important;
    min-width: 0;
}

#hero {
    text-align: center;
    padding: 20px 0 12px;
}
#hero h1 {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#hero p {
    color: #64748b;
    margin: 4px 0 0;
    font-size: 0.9rem;
}

#content-wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 24px;
}

#footer {
    text-align: center;
    padding: 12px 0 8px;
    color: #94a3b8;
    font-size: 0.75rem;
}

#download-btn {
    min-height: 48px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
}

#send-btn, #upload-btn, #cancel-btn {
    min-height: 42px !important;
    font-weight: 600 !important;
}

/* Live progress bar */
#progress-bar-container { margin: 12px 0 8px; }
#progress-bar-container .progress-track {
    width: 100%; height: 10px; background: #e2e8f0;
    border-radius: 999px; overflow: hidden;
}
#progress-bar-container .progress-fill {
    height: 100%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 999px; transition: width 0.4s ease;
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(79,70,229,0.3); }
    50% { box-shadow: 0 0 8px 2px rgba(79,70,229,0.3); }
}
#progress-bar-container .progress-fill.active {
    animation: pulse-glow 1.5s ease-in-out infinite;
}

/* Chat selector styling */
#chat-selector {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#chat-selector label { display: none !important; }
#chat-selector .wrap {
    background: transparent !important;
    gap: 4px !important;
}
#chat-selector input[type="radio"] { display: none !important; }
#chat-selector label span {
    display: block;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    color: #e2e8f0 !important;
    font-size: 0.85rem;
    transition: background 0.15s;
    background: transparent;
}
#chat-selector label span:hover {
    background: rgba(255,255,255,0.08);
}
#chat-selector input[type="radio"]:checked + span {
    background: rgba(99,102,241,0.3);
}
"""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(rag_engine: RAGEngine, handbook_gen: HandbookGenerator) -> gr.Blocks:
    """Create the Gradio app with sidebar and full-width layout."""

    # Pre-compute sidebar once at build time (lazy-loaded on interactions)
    _cached_choices = _chat_choices(_list_chats())

    with gr.Blocks(title="PaperLens") as app:

        # Session state — each browser tab gets its own state
        current_chat_id = gr.State(value=lambda: f"chat_{int(time.time())}")
        session_uploads = gr.State(value=[])  # track uploaded files per session

        with gr.Row(elem_id="main-row", equal_height=True):

            # ── Sidebar (hidden until PDFs indexed) ──
            with gr.Column(elem_id="sidebar", scale=0, min_width=260, visible=False) as sidebar_section:
                with gr.Column(elem_id="sidebar-inner"):
                    gr.HTML(
                        '<div id="sidebar-header">'
                        '<h3>Conversations</h3>'
                        '</div>'
                    )
                    new_chat_btn = gr.Button("+ New Chat", elem_id="new-chat-btn", visible=False)
                    chat_selector = gr.Radio(
                        choices=[],
                        label=None,
                        show_label=False,
                        elem_id="chat-selector",
                    )

            # ── Main content ──
            with gr.Column(elem_id="main-content", scale=1):

                gr.HTML(
                    '<div id="hero">'
                    "<h1>PaperLens</h1>"
                    "<p>Upload PDFs, ask questions, generate 20,000+ word handbooks</p>"
                    "</div>"
                )

                with gr.Column(elem_id="content-wrapper"):

                    # ── Phase 1: Upload (visible by default) ──
                    with gr.Column(visible=True) as upload_section:
                        gr.HTML(
                            '<div style="padding:0 0 12px;">'
                            '<h2 style="font-size:1.3rem;font-weight:700;margin:0;'
                            'background:linear-gradient(135deg,#4f46e5,#7c3aed);'
                            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                            'background-clip:text;">Upload Documents</h2>'
                            '</div>'
                        )
                        file_upload = gr.File(
                            label=None,
                            show_label=False,
                            file_types=[".pdf"],
                            file_count="multiple",
                            type="filepath",
                        )
                        with gr.Row():
                            index_btn = gr.Button(
                                "Index Documents", variant="primary", elem_id="upload-btn",
                                scale=3, visible=False,
                            )
                            cancel_btn = gr.Button(
                                "Cancel", variant="secondary", elem_id="cancel-btn",
                                scale=1, visible=False,
                            )
                        progress_bar = gr.HTML("", elem_id="progress-bar-container", visible=False)
                        upload_status = gr.Markdown("")

                    # ── Phase 2: Chat (hidden until upload) ──
                    with gr.Column(visible=False) as chat_section:
                        chatbot = gr.Chatbot(
                            height=480,
                            show_label=False,
                            avatar_images=(
                                None,
                                "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=handbook&backgroundColor=6366f1",
                            ),
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask a question, or type 'Create a handbook on [topic]'...",
                                show_label=False,
                                scale=5,
                                lines=1,
                                submit_btn=False,
                                autofocus=True,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

                        download_pdf = gr.DownloadButton(
                            label="Download Handbook",
                            visible=False,
                            variant="primary",
                            elem_id="download-btn",
                        )

                gr.HTML(
                    '<div id="footer">'
                    "PaperLens &bull; Grok 4.1 &bull; LightRAG &bull; Supabase"
                    "</div>"
                )

        # ── Wiring ──

        def _progress_html(frac: float) -> str:
            pct = max(0, min(100, int(frac * 100)))
            active = "active" if pct < 100 else ""
            return (
                f'<div class="progress-track">'
                f'<div class="progress-fill {active}" style="width:{pct}%"></div>'
                f'</div>'
            )

        async def process_upload(files, uploads):
            if not files:
                yield gr.update(), "Please select PDF files to upload.", gr.update(), gr.update(), gr.update(), uploads, gr.update(), gr.update(), gr.update()
                return
            # Track uploaded files in session
            new_uploads = uploads or []
            for f in files:
                import os as _os
                new_uploads.append({
                    "name": _os.path.basename(f),
                    "path": f,
                    "uploaded_at": time.strftime("%H:%M:%S"),
                })
            async for status_text, frac in handle_upload(files, rag_engine):
                is_done = frac >= 1.0
                yield (
                    gr.update(value=_progress_html(frac), visible=not is_done),
                    status_text,
                    gr.update(visible=False) if is_done else gr.update(),
                    gr.update(visible=True) if is_done else gr.update(),
                    gr.update(visible=False) if is_done else gr.update(),
                    new_uploads,
                    gr.update(choices=_chat_choices(_list_chats()), value=None) if is_done else gr.update(),
                    gr.update(visible=True) if is_done else gr.update(),
                    # Show entire sidebar after indexing
                    gr.update(visible=True) if is_done else gr.update(),
                )

        file_upload.change(
            fn=lambda files: gr.update(visible=bool(files)),
            inputs=[file_upload],
            outputs=[index_btn],
        )

        def on_index_start():
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, value=_progress_html(0))

        index_event = index_btn.click(
            fn=on_index_start,
            outputs=[index_btn, cancel_btn, progress_bar],
        ).then(
            fn=process_upload,
            inputs=[file_upload, session_uploads],
            outputs=[progress_bar, upload_status, upload_section, chat_section, cancel_btn,
                     session_uploads, chat_selector, new_chat_btn, sidebar_section],
        )

        cancel_btn.click(
            fn=None,
            cancels=[index_event],
        ).then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "Indexing cancelled."),
            outputs=[index_btn, cancel_btn, progress_bar, upload_status],
        )

        # ── Chat wiring with session persistence ──

        async def on_chat(message, history, chat_id):
            async for h, dl in handle_chat(message, history, rag_engine, handbook_gen):
                if h:
                    _save_chat(chat_id, h)
                chats = _list_chats()
                yield h, dl, gr.update(choices=_chat_choices(chats))

        stored_msg = gr.State("")

        def save_and_clear(message):
            return message, ""

        send_btn.click(
            fn=save_and_clear,
            inputs=[msg_input],
            outputs=[stored_msg, msg_input],
        ).then(
            fn=on_chat,
            inputs=[stored_msg, chatbot, current_chat_id],
            outputs=[chatbot, download_pdf, chat_selector],
            show_progress="minimal",
        )

        msg_input.submit(
            fn=save_and_clear,
            inputs=[msg_input],
            outputs=[stored_msg, msg_input],
        ).then(
            fn=on_chat,
            inputs=[stored_msg, chatbot, current_chat_id],
            outputs=[chatbot, download_pdf, chat_selector],
            show_progress="minimal",
        )

        # ── New chat button ──
        def start_new_chat():
            new_id = f"chat_{int(time.time())}"
            chats = _list_chats()
            return [], new_id, gr.update(choices=_chat_choices(chats), value=None)

        new_chat_btn.click(
            fn=start_new_chat,
            outputs=[chatbot, current_chat_id, chat_selector],
        )

        # ── Load chat from sidebar radio ──
        def load_chat_from_selector(selected_label):
            if not selected_label:
                return gr.update(), gr.update()
            chats = _list_chats()
            id_map = _chat_id_map(chats)
            chat_id = id_map.get(selected_label, "")
            if not chat_id:
                return gr.update(), gr.update()
            data = _load_chat(chat_id)
            if not data:
                return gr.update(), gr.update()
            return data["messages"], chat_id

        chat_selector.change(
            fn=load_chat_from_selector,
            inputs=[chat_selector],
            outputs=[chatbot, current_chat_id],
        )

    return app
