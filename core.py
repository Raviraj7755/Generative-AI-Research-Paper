"""
core.py
--------
Implements the Hybrid Research Agent architecture described in:
"Generative AI Agents: Architecting Autonomous Intelligence for the Real World"

Maps directly onto the paper's Fig. 1 (Proposed Hybrid Architecture) and
Fig. 2 (Workflow of the Proposed Research Agent):

    1. User                      -> Streamlit UI (app.py)
    2. Upload research papers     -> DocumentProcessor.load_pdfs()
    3. PDF processing/extraction  -> DocumentProcessor.extract_text()
    4. Document loader/indexing   -> VectorStore.build_index()
    5. User question               -> app.py chat input
    6. RAG retrieval               -> VectorStore.retrieve()
    7. Context construction        -> ResearchAgent._build_context()
    8. Gemini LLM generation       -> GeminiGenerator.generate()
    9. Reasoning & memory          -> ReasoningPlanner + StructuredMemory
    10. Verification (self-check)  -> SelfVerifier.verify()
    11. Final answer               -> ResearchAgent.answer_query()

Every class below is intentionally kept simple and readable so it can be
explained line-by-line in a viva/presentation.
"""

import os
import re
import json
import time
import uuid
import hashlib
import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai


# --------------------------------------------------------------------------
# 1 & 2. DOCUMENT PROCESSOR  (Fig 2, steps 2-4: upload -> extract -> chunk)
# --------------------------------------------------------------------------
class DocumentProcessor:
    """Handles PDF upload, text extraction, and chunking into passages."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_pages(self, file):
        """Extract text per PDF page, so every chunk can cite a page number."""
        reader = PdfReader(file)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append({"page": i, "text": self._clean(page_text)})
        return pages

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def chunk_document(self, pages, source: str):
        """Chunk each page independently so page numbers stay accurate."""
        chunks = []
        for page in pages:
            text = page["text"]
            if not text:
                continue
            start = 0
            n = len(text)
            while start < n:
                end = min(start + self.chunk_size, n)
                chunk = text[start:end]
                chunks.append({"text": chunk, "source": source, "page": page["page"]})
                start += self.chunk_size - self.chunk_overlap
        return chunks


# --------------------------------------------------------------------------
# CHAT HISTORY STORE — persists conversations across app restarts
# --------------------------------------------------------------------------
class ChatHistoryStore:
    """Saves and lists past chat sessions on disk, so users can start a new
    chat, revisit an old one, or delete one — independent of the Gemini
    pipeline (pure local file storage)."""

    HISTORY_DIR = "chat_history"

    def __init__(self):
        os.makedirs(self.HISTORY_DIR, exist_ok=True)

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    def _path(self, session_id: str) -> str:
        return os.path.join(self.HISTORY_DIR, f"{session_id}.json")

    def list_sessions(self):
        """Return all saved sessions, most recent first."""
        sessions = []
        for fname in os.listdir(self.HISTORY_DIR):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.HISTORY_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            sessions.append({
                "id": fname[:-5],
                "title": data.get("title") or "Untitled chat",
                "mtime": os.path.getmtime(path),
            })
        sessions.sort(key=lambda s: s["mtime"], reverse=True)
        return sessions

    def load_session(self, session_id: str):
        path = self._path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])

    def save_session(self, session_id: str, messages):
        """Save (or overwrite) a session's full message list. Called after
        every turn so nothing is lost if the app closes unexpectedly."""
        if not messages:
            return
        first_user_msg = next((m["text"] for m in messages if m["role"] == "user"), "Untitled chat")
        title = first_user_msg[:60] + ("..." if len(first_user_msg) > 60 else "")
        with open(self._path(session_id), "w", encoding="utf-8") as f:
            json.dump({"title": title, "messages": messages}, f, ensure_ascii=False)

    def delete_session(self, session_id: str):
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)


# --------------------------------------------------------------------------
# 3. VECTOR STORE / RAG RETRIEVAL MODULE (Fig 2, steps 4 & 6; Fig 1 module 1)
# --------------------------------------------------------------------------
class VectorStore:
    """FAISS-based vector index over document chunks, embedded via Gemini."""

    EMBED_MODEL = "models/gemini-embedding-001"

    def __init__(self):
        self.index = None
        self.chunks = []   # kept in sync for compatibility with retrieve()/tools
        self.entries = []  # list of {"chunk": {...}, "vector": np.array} — enables deletion
        self.dim = None

    def _embed(self, texts, batch_size: int = 90, progress_cb=None):
        """Call Gemini embedding API in batches, with automatic retry on the
        free-tier rate limit (429 ResourceExhausted), and a short pause
        between batches to avoid tripping the per-minute limit at all.

        progress_cb, if given, is called as progress_cb(batch_num, n_batches)
        after each batch completes, so the UI can show real progress instead
        of sitting frozen until the whole thing finishes.
        """
        vectors = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[i:i + batch_size]
            batch_vectors = self._embed_batch_with_retry(batch)
            vectors.extend(batch_vectors)
            if progress_cb:
                progress_cb(batch_num, n_batches)
            if n_batches > 1 and i + batch_size < len(texts):
                time.sleep(2)  # small pacing gap between batches
        return np.array(vectors, dtype="float32")

    @staticmethod
    def _embed_batch_with_retry(batch, max_retries: int = 5):
        """Retries with backoff if the API returns a rate-limit (429) error."""
        import time
        from google.api_core.exceptions import ResourceExhausted

        for attempt in range(max_retries):
            try:
                result = genai.embed_content(model=VectorStore.EMBED_MODEL, content=batch)
                batch_vectors = result["embedding"]
                if batch_vectors and isinstance(batch_vectors[0], float):
                    batch_vectors = [batch_vectors]
                return batch_vectors
            except ResourceExhausted:
                wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s, 40s, 50s
                time.sleep(wait_time)
        # Final attempt without catching, so the real error surfaces if still failing
        result = genai.embed_content(model=VectorStore.EMBED_MODEL, content=batch)
        batch_vectors = result["embedding"]
        if batch_vectors and isinstance(batch_vectors[0], float):
            batch_vectors = [batch_vectors]
        return batch_vectors

    def build_index(self, chunks):
        """Embed all chunks and build/refresh the FAISS index (used for a
        fresh, from-scratch index)."""
        self.entries = []
        self.chunks = []
        self.index = None
        if not chunks:
            return
        vectors = self._embed([c["text"] for c in chunks])
        self.add(chunks, vectors, already_normalized=False)

    def add(self, chunks, vectors, already_normalized=False):
        """Add pre-computed chunks + embedding vectors to the index without
        re-calling the embedding API. This is what makes cached documents
        free to reload."""
        if len(chunks) == 0:
            return
        vectors = np.array(vectors, dtype="float32")
        if not already_normalized:
            faiss.normalize_L2(vectors)
        for chunk, vec in zip(chunks, vectors):
            self.entries.append({"chunk": chunk, "vector": vec})
        self._rebuild_index()

    def remove_source(self, source_name: str):
        """Remove every chunk belonging to a given document (by filename)
        and rebuild the index — used when a document is deleted."""
        self.entries = [e for e in self.entries if e["chunk"]["source"] != source_name]
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the FAISS index from the current set of entries. Cheap
        enough at prototype scale (hundreds of chunks) to redo on every
        add/remove, and keeps deletion simple and correct."""
        if not self.entries:
            self.index = None
            self.chunks = []
            self.dim = None
            return
        vectors = np.array([e["vector"] for e in self.entries], dtype="float32")
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.chunks = [e["chunk"] for e in self.entries]

    def retrieve(self, query: str, top_k: int = 4):
        """Return the top_k most relevant chunks for a query."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q_vec = self._embed([query])
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(score)
            results.append(item)
        return results


# --------------------------------------------------------------------------
# 4. STRUCTURED MEMORY MODULE (Fig 1: "Structured Memory")
# --------------------------------------------------------------------------
class StructuredMemory:
    """
    Keeps conversational history and short-term facts, so the agent can
    refer back to earlier turns (addresses the paper's "limited long-term
    memory" problem within a session).
    """

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.history = []   # list of {"role": "user"/"agent", "text": ...}
        self.facts = {}      # key facts extracted across the session

    def add_turn(self, role: str, text: str):
        self.history.append({"role": role, "text": text})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def add_fact(self, key: str, value: str):
        self.facts[key] = value

    def as_context(self) -> str:
        if not self.history:
            return "No prior conversation."
        lines = [f'{t["role"]}: {t["text"]}' for t in self.history[-6:]]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 5. REASONING & PLANNING MODULE (Fig 1, module 2)
# --------------------------------------------------------------------------
class ReasoningPlanner:
    """
    Decides HOW to answer a query before retrieval happens:
    - is this a follow-up needing memory?
    - does it need document retrieval?
    - does it need an external tool (e.g. word/page count)?
    This is a lightweight rule-based planner (kept simple deliberately -
    the paper's LLM-based reasoning can be swapped in later).
    """

    TOOL_TRIGGERS = {
        "word count": "word_count",
        "how many pages": "page_count",
        "how many chunks": "chunk_count",
    }

    def plan(self, query: str, memory: StructuredMemory):
        plan = {
            "needs_retrieval": True,
            "needs_memory": len(memory.history) > 0,
            "tool": None,
        }
        q_lower = query.lower()
        for trigger, tool_name in self.TOOL_TRIGGERS.items():
            if trigger in q_lower:
                plan["tool"] = tool_name
                plan["needs_retrieval"] = False
        return plan


# --------------------------------------------------------------------------
# 6. TOOL ORCHESTRATION (Fig 1: "External Tools")
# --------------------------------------------------------------------------
class ToolOrchestrator:
    """A tiny set of deterministic tools the agent can call instead of
    hallucinating an answer (demonstrates the "intelligent tool
    orchestration" idea from the paper without needing external APIs)."""

    def run(self, tool_name: str, vector_store: VectorStore):
        if tool_name == "word_count":
            total_words = sum(len(c["text"].split()) for c in vector_store.chunks)
            return f"The uploaded documents contain approximately {total_words} words."
        if tool_name == "page_count":
            return f"There are {len(set(c['source'] for c in vector_store.chunks))} document(s) loaded."
        if tool_name == "chunk_count":
            return f"The retrieval index currently holds {len(vector_store.chunks)} chunks."
        return "Tool not recognized."


# --------------------------------------------------------------------------
# 7. GEMINI GENERATION MODULE (Fig 1, module 3; Fig 2, step 8)
# --------------------------------------------------------------------------
class GeminiGenerator:
    """Wraps the Gemini chat model for grounded answer generation."""

    def __init__(self, model_name: str = "gemini-3.1-flash-lite"):
        self.model = genai.GenerativeModel(model_name)

    def generate(self, query: str, context: str, memory_context: str) -> str:
        prompt = f"""You are a focused research assistant. Follow these rules strictly:

1. Answer ONLY using facts found in the RETRIEVED CONTEXT below. Do not add
   outside knowledge. If the context does not contain the answer, say so
   honestly instead of guessing.
2. Stay strictly on-topic — do not wander into related-but-unasked subtopics.
   Answer exactly what was asked, concisely.
3. Reply in the SAME LANGUAGE the question was asked in (the question may be
   in any language, e.g. Hindi, Marathi, English, etc.). Match that language.
4. After each factual claim, cite its source in this exact format:
   (Source: <document name>, Page <page number>)
   If a claim draws on multiple chunks, cite all of them.

--- CONVERSATION MEMORY ---
{memory_context}

--- RETRIEVED CONTEXT (each block shows its source document and page) ---
{context}

--- QUESTION ---
{query}

Now answer, following all four rules above."""
        response = self.model.generate_content(prompt)
        return response.text


# --------------------------------------------------------------------------
# 8. SELF-VERIFICATION MODULE (Fig 1 & 2, step 10)
# --------------------------------------------------------------------------
class SelfVerifier:
    """
    Runs a second, independent Gemini pass whose only job is to check the
    generated answer against the retrieved context and flag unsupported
    claims (this is what reduces hallucination per the paper's design).
    """

    def __init__(self, model_name: str = "gemini-3.1-flash-lite"):
        self.model = genai.GenerativeModel(model_name)

    def verify(self, answer: str, context: str) -> dict:
        prompt = f"""You are a strict fact-checker. Compare the ANSWER to the
CONTEXT. Reply in this exact JSON format and nothing else:
{{"grounded": true/false, "note": "one short sentence explaining why"}}

CONTEXT:
{context}

ANSWER:
{answer}"""
        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip().strip("```json").strip("```").strip()
            result = json.loads(raw)
            return result
        except Exception:
            return {"grounded": None, "note": "Verification could not be parsed."}


# --------------------------------------------------------------------------
# ORCHESTRATOR: full Fig. 2 pipeline, steps 1-11
# --------------------------------------------------------------------------
class ResearchAgent:
    DOCS_DIR = "uploaded_docs"     # stores original PDF files, persists across runs
    CACHE_DIR = "doc_cache"        # stores extracted chunks + embeddings per file

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.memory = StructuredMemory()
        self.planner = ReasoningPlanner()
        self.tools = ToolOrchestrator()
        self.generator = GeminiGenerator()
        self.verifier = SelfVerifier()
        os.makedirs(self.DOCS_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.loaded_hashes = set()  # hashes already added to this session's index
        self._autoload_cached_library()

    def _autoload_cached_library(self):
        """On startup, silently load every document that was already
        processed before (from cache — zero API calls) so the whole library
        is available in chat immediately, with no extra click needed."""
        for doc in self.list_library():
            if doc["cached"]:
                self.load_from_library(doc["hash"])

    @staticmethod
    def _file_hash(file_bytes: bytes) -> str:
        return hashlib.md5(file_bytes).hexdigest()

    def list_library(self):
        """List every document ever uploaded (persists across app restarts),
        newest first."""
        docs = []
        for fname in os.listdir(self.DOCS_DIR):
            path = os.path.join(self.DOCS_DIR, fname)
            if "__" in fname:
                file_hash, original_name = fname.split("__", 1)
            else:
                file_hash, original_name = fname, fname
            docs.append({
                "hash": file_hash,
                "filename": original_name,
                "path": path,
                "mtime": os.path.getmtime(path),
                "cached": os.path.exists(os.path.join(self.CACHE_DIR, f"{file_hash}.json")),
            })
        docs.sort(key=lambda d: d["mtime"], reverse=True)
        return docs

    def process_uploaded_file(self, uploaded_file, progress_cb=None):
        """Process one uploaded file: reuse the cached embeddings if this
        exact file was ever processed before (zero API calls), otherwise
        extract + embed it fresh and cache the result for next time.

        progress_cb, if given, is called as progress_cb(stage: str, frac: float)
        at each meaningful step so the UI can show real, moving progress
        instead of a bar frozen until the whole call returns.
        """
        file_bytes = uploaded_file.getvalue()
        file_hash = self._file_hash(file_bytes)
        cache_path = os.path.join(self.CACHE_DIR, f"{file_hash}.json")
        stored_path = os.path.join(self.DOCS_DIR, f"{file_hash}__{uploaded_file.name}")

        if not os.path.exists(stored_path):
            with open(stored_path, "wb") as f:
                f.write(file_bytes)

        if os.path.exists(cache_path):
            if progress_cb:
                progress_cb("Loading cached embeddings...", 0.9)
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            chunks = cached["chunks"]
            vectors = np.array(cached["embeddings"], dtype="float32")
            from_cache = True
        else:
            if progress_cb:
                progress_cb("Extracting text from PDF...", 0.05)
            uploaded_file.seek(0)
            pages = self.doc_processor.extract_pages(uploaded_file)

            if progress_cb:
                progress_cb("Chunking document...", 0.15)
            chunks = self.doc_processor.chunk_document(pages, source=uploaded_file.name)

            def _embed_progress(batch_num, n_batches):
                if progress_cb:
                    frac = 0.15 + 0.8 * (batch_num / n_batches)
                    progress_cb(f"Embedding chunks (batch {batch_num}/{n_batches})...", frac)

            vectors = (
                self.vector_store._embed([c["text"] for c in chunks], progress_cb=_embed_progress)
                if chunks else np.zeros((0, 1))
            )
            if progress_cb:
                progress_cb("Saving to cache...", 0.97)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunks, "embeddings": vectors.tolist()}, f)
            from_cache = False

        if file_hash not in self.loaded_hashes and chunks:
            self.vector_store.add(chunks, vectors)
            self.loaded_hashes.add(file_hash)

        return {"n_chunks": len(chunks), "from_cache": from_cache, "hash": file_hash}

    def load_from_library(self, file_hash: str):
        """Load an already-processed document from the library into the
        active index, purely from cache (no API calls at all)."""
        cache_path = os.path.join(self.CACHE_DIR, f"{file_hash}.json")
        if not os.path.exists(cache_path) or file_hash in self.loaded_hashes:
            return 0
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        chunks = cached["chunks"]
        vectors = np.array(cached["embeddings"], dtype="float32")
        if chunks:
            self.vector_store.add(chunks, vectors)
            self.loaded_hashes.add(file_hash)
        return len(chunks)

    def load_documents(self, uploaded_files):
        """Kept for backward compatibility: process a batch of files."""
        total = 0
        for f in uploaded_files:
            result = self.process_uploaded_file(f)
            total += result["n_chunks"]
        return total

    def delete_document(self, file_hash: str):
        """Permanently delete a document: removes the stored PDF, its cache
        file, and (if currently loaded) its chunks from the active index."""
        removed_filename = None
        for fname in os.listdir(self.DOCS_DIR):
            if fname.startswith(file_hash + "__") or fname == file_hash:
                removed_filename = fname.split("__", 1)[-1]
                os.remove(os.path.join(self.DOCS_DIR, fname))

        cache_path = os.path.join(self.CACHE_DIR, f"{file_hash}.json")
        if os.path.exists(cache_path):
            os.remove(cache_path)

        if file_hash in self.loaded_hashes and removed_filename:
            self.vector_store.remove_source(removed_filename)
            self.loaded_hashes.discard(file_hash)

        return removed_filename

    @staticmethod
    def _build_context(retrieved_chunks):
        if not retrieved_chunks:
            return "No relevant context retrieved."
        return "\n\n".join(
            f"[Source: {c['source']} | Page {c.get('page', '?')} | relevance {c['score']:.2f}]\n{c['text']}"
            for c in retrieved_chunks
        )

    def answer_query(self, query: str, top_k: int = 4):
        """
        Runs the full pipeline and returns a dict with every intermediate
        artifact so the UI can visualize each stage of Fig. 2 live.
        """
        trace = {}

        # Step 9 (planning happens first so retrieval can be skipped for tools)
        plan = self.planner.plan(query, self.memory)
        trace["plan"] = plan

        if plan["tool"]:
            answer = self.tools.run(plan["tool"], self.vector_store)
            trace["retrieved"] = []
            trace["context"] = "N/A (tool used instead of retrieval)"
            trace["verification"] = {"grounded": True, "note": "Deterministic tool output."}
            self.memory.add_turn("user", query)
            self.memory.add_turn("agent", answer)
            trace["answer"] = answer
            return trace

        # Step 6: RAG retrieval
        retrieved = self.vector_store.retrieve(query, top_k=top_k)
        trace["retrieved"] = retrieved

        # Step 7: context construction
        context = self._build_context(retrieved)
        trace["context"] = context

        # Step 8: Gemini generation (grounded in context + memory)
        memory_context = self.memory.as_context()
        answer = self.generator.generate(query, context, memory_context)
        trace["answer"] = answer

        # Step 10: self-verification
        verification = self.verifier.verify(answer, context)
        trace["verification"] = verification

        # Update structured memory
        self.memory.add_turn("user", query)
        self.memory.add_turn("agent", answer)

        return trace
