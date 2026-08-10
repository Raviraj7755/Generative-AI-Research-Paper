"""
app.py
------
Streamlit front-end for the prototype Research Agent described in the report.
Run with:  streamlit run app.py
"""

import base64
import streamlit as st
from core import ResearchAgent, ChatHistoryStore

st.set_page_config(page_title="Research Agent Prototype", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Session + chat history bootstrap
# ---------------------------------------------------------------------------
if "history_store" not in st.session_state:
    st.session_state.history_store = ChatHistoryStore()
if "session_id" not in st.session_state:
    st.session_state.session_id = st.session_state.history_store.new_session_id()
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# FIX: the document library list is now cached across reruns and only
# recomputed when a document is actually added or deleted. Previously
# `agent.list_library()` ran on EVERY script rerun (which Streamlit
# triggers on almost any interaction — opening an expander, sending a
# chat message, clicking history), so any hashing/disk work inside it
# re-ran constantly, causing the "click and it freezes for 10-15s" feel.
if "library_dirty" not in st.session_state:
    st.session_state.library_dirty = True
if "library_cache" not in st.session_state:
    st.session_state.library_cache = []

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; font-size: 2rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); margin-top: 0.4rem; font-size: 0.95rem; }
    .stChatMessage { border-radius: 12px; }
    .source-card {
        background: rgba(124, 58, 237, 0.08);
        border-left: 3px solid #7C3AED;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .pill-green { background: #DCFCE7; color: #166534; }
    .pill-red { background: #FEE2E2; color: #991B1B; }
    .pill-gray { background: #F3F4F6; color: #374151; }
    .doc-row {
        padding: 0.4rem 0.6rem;
        border-radius: 8px;
        margin-bottom: 0.3rem;
        background: rgba(128,128,128,0.06);
    }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.15); }
    /* Icon-only delete buttons: small natural square instead of a stretched block */
    div[data-testid="column"]:has(button[title]) button {
        width: 2.4rem !important;
        min-width: 2.4rem !important;
        height: 2.4rem;
        padding: 0 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧠 Generative AI Research Agent</h1>
    <p>Hybrid Architecture Prototype — RAG · Structured Memory · Reasoning &amp; Planning · Tool Orchestration · Self-Verification</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached PDF -> base64 (this used to re-read + re-encode every PDF in the
# library on EVERY rerun, which is what was causing the constant lag/dim).
# Keyed on content hash, so it only ever runs once per unique document.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_pdf_data_uri(path: str, doc_hash: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


# ---------------------------------------------------------------------------
# Sidebar: API key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    api_key = st.text_input("Gemini API Key", type="password", help="Get one at aistudio.google.com/apikey")

    if api_key and "agent" not in st.session_state:
        st.session_state.agent = ResearchAgent(api_key)

    if "agent" not in st.session_state:
        st.warning("👈 Enter your Gemini API key to begin.")
        st.stop()

    agent = st.session_state.agent

    st.divider()
    st.markdown("### 📄 Upload a Document")

    # Isolated in a fragment: uploading/processing a document now only
    # reruns this block, not the whole page (chat history, doc list, etc.)
    @st.fragment
    def upload_section():
        uploaded_file = st.file_uploader(
            "Upload a single PDF", type=["pdf"], accept_multiple_files=False, label_visibility="collapsed"
        )

        if uploaded_file and st.button("🚀 Process Document", use_container_width=True, type="primary"):
            progress = st.progress(0, text="Checking cache...")

            def on_progress(message: str, frac: float):
                progress.progress(min(int(frac * 100), 100), text=message)

            result = agent.process_uploaded_file(uploaded_file, progress_cb=on_progress)
            progress.progress(100, text="Done!")
            if result["from_cache"]:
                st.success(f"✅ Loaded from cache — {result['n_chunks']} chunks (no API calls used).")
            else:
                st.success(f"✅ Processed and cached — {result['n_chunks']} chunks.")
            st.session_state.docs_loaded = True
            # FIX: mark the library cache stale so it recomputes exactly
            # once on the next rerun — not on every future interaction.
            st.session_state.library_dirty = True
            # Full rerun needed so the doc list / chunk count outside this
            # fragment picks up the new document.
            st.rerun()

    upload_section()

    st.divider()

    # FIX: only call the (potentially expensive) agent.list_library() when
    # something has actually changed, instead of on every rerun.
    if st.session_state.library_dirty:
        st.session_state.library_cache = agent.list_library()
        st.session_state.library_dirty = False
    library = st.session_state.library_cache

    with st.expander(f"📁 Uploaded Documents ({len(library)})", expanded=False):
        st.caption("Click a document to open the full PDF in a new tab. All processed documents are automatically part of the chat's knowledge base.")
        if not library:
            st.caption("No documents uploaded yet.")
        else:
            for doc in library:
                try:
                    href = get_pdf_data_uri(doc["path"], doc["hash"])
                    row = st.columns([6, 1])
                    with row[0]:
                        st.markdown(
                            f'<a href="{href}" target="_blank" style="display:block; padding:0.5rem 0.9rem; '
                            f'background:rgba(128,128,128,0.10); border-radius:8px; text-decoration:none; '
                            f'color:inherit; font-size:0.88rem; white-space:nowrap; overflow:hidden; '
                            f'text-overflow:ellipsis;">📄 {doc["filename"]}</a>',
                            unsafe_allow_html=True,
                        )
                    with row[1]:
                        if st.button("🗑️", key=f"delete_{doc['hash']}", help=f"Delete {doc['filename']}"):
                            removed_name = agent.delete_document(doc["hash"])
                            get_pdf_data_uri.clear()  # drop the cached b64 for the deleted file
                            # FIX: same invalidation on delete.
                            st.session_state.library_dirty = True
                            st.toast(f"Deleted {removed_name}")
                            st.rerun()
                except Exception as e:
                    st.caption(f"Could not load {doc['filename']}: {e}")

    n_total = len(agent.vector_store.chunks)
    if n_total:
        st.info(f"🧩 {n_total} chunks currently active in this chat's knowledge base.")

    st.divider()
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.history_store.save_session(st.session_state.session_id, st.session_state.chat_log)
        st.session_state.session_id = st.session_state.history_store.new_session_id()
        st.session_state.chat_log = []
        st.rerun()

    sessions = st.session_state.history_store.list_sessions()
    with st.expander(f"🕘 Conversation History ({len(sessions)})", expanded=False):
        if not sessions:
            st.caption("No past conversations yet.")
        else:
            for s in sessions:
                row = st.columns([6, 1])
                with row[0]:
                    active = " ✅" if s["id"] == st.session_state.session_id else ""
                    if st.button(s["title"] + active, key=f"hist_{s['id']}", use_container_width=True):
                        st.session_state.history_store.save_session(st.session_state.session_id, st.session_state.chat_log)
                        st.session_state.session_id = s["id"]
                        st.session_state.chat_log = st.session_state.history_store.load_session(s["id"])
                        st.rerun()
                with row[1]:
                    if st.button("🗑️", key=f"histdel_{s['id']}", help="Delete this conversation"):
                        st.session_state.history_store.delete_session(s["id"])
                        if s["id"] == st.session_state.session_id:
                            st.session_state.session_id = st.session_state.history_store.new_session_id()
                            st.session_state.chat_log = []
                        st.rerun()

    st.divider()
    st.caption("💡 Ask in any language — Hindi, Marathi, English, etc. Answers cite exact source + page number.")

# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------
for turn in st.session_state.chat_log:
    avatar = "🧑" if turn["role"] == "user" else "🧠"
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["text"])

query = st.chat_input("Ask a question about the uploaded papers (any language)...")

if query:
    st.session_state.chat_log.append({"role": "user", "text": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🧠"):
        pipeline_box = st.status("Running Hybrid Agent Pipeline...", expanded=True)

        pipeline_box.write("**Step 9 — Reasoning & Planning:** deciding retrieval vs. tool use...")
        # FIX: this single call runs retrieval + generation + verification
        # synchronously — nothing else can update pipeline_box until it
        # returns, which is why it looked frozen for the full 20-25s.
        # This note at least tells the viewer what's happening during that
        # gap instead of leaving it silent. A true fix (streaming per-step
        # updates) needs a progress_cb hook added inside core.py's
        # answer_query, similar to process_uploaded_file's progress_cb.
        pipeline_box.write("_(retrieval → generation → self-verification run sequentially here — ~20s on the free-tier API)_")
        trace = st.session_state.agent.answer_query(query)

        plan = trace["plan"]
        if plan["tool"]:
            pipeline_box.write(f"→ Planner routed this to tool: `{plan['tool']}` (RAG skipped).")
        else:
            pipeline_box.write("→ Planner routed this to RAG retrieval.")
            pipeline_box.write(f"**Step 6 — RAG Retrieval:** fetched {len(trace['retrieved'])} chunk(s).")
            pipeline_box.write("**Step 7 — Context Construction:** merged chunks + memory into prompt.")
            pipeline_box.write("**Step 8 — Gemini Generation:** produced grounded, cited answer.")

        v = trace["verification"]
        if v.get("grounded") is True:
            pill = '<span class="status-pill pill-green">✅ Grounded</span>'
        elif v.get("grounded") is False:
            pill = '<span class="status-pill pill-red">⚠️ Check needed</span>'
        else:
            pill = '<span class="status-pill pill-gray">❔ Unverified</span>'
        pipeline_box.markdown(f"**Step 10 — Self-Verification:** {pill} — {v.get('note', '')}", unsafe_allow_html=True)
        pipeline_box.update(label="✅ Pipeline complete", state="complete", expanded=False)

        st.markdown(trace["answer"])
        st.session_state.chat_log.append({"role": "assistant", "text": trace["answer"]})

        if trace["retrieved"]:
            with st.expander("🔍 Retrieved source chunks (transparency)"):
                for c in trace["retrieved"]:
                    st.markdown(
                        f"""<div class="source-card">
                        <b>{c['source']}</b> — Page {c.get('page', '?')} — relevance {c['score']:.2f}<br>
                        <span style="opacity:0.8; font-size:0.9rem;">{c['text'][:350]}{'...' if len(c['text']) > 350 else ''}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.session_state.history_store.save_session(st.session_state.session_id, st.session_state.chat_log)