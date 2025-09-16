import os
import io
import tempfile
import pickle
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader

from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# Config
# ----------------------------
load_dotenv()  # Load environment variables from .env

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = "gpt-4o-mini"

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    order: int

@dataclass
class IndexBundle:
    index: Any
    dim: int
    chunks: List[Chunk]
    model: str

# ----------------------------
# Utilities
# ----------------------------
def get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def embed_texts(client: OpenAI, texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    out = []
    B = 128
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    return np.vstack(out) if out else np.zeros((0, 1536), dtype=np.float32)

def pdf_to_pages(file: io.BytesIO) -> List[str]:
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages

def simple_txt_read(file: io.BytesIO) -> str:
    raw = file.read()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("latin-1", errors="ignore")

def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        window = text[start:end]
        back = window.rfind(". ")
        if back > 250:
            end = start + back + 1
            window = text[start:end]
        chunks.append(window.strip())
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def build_corpus(files: List[Dict[str, Any]], *, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for f in files:
        name = f["name"]
        data = f["data"]

        if name.lower().endswith(".pdf"):
            pages = pdf_to_pages(io.BytesIO(data))
            for page_i, page_text in enumerate(pages, start=1):
                for order, c in enumerate(chunk_text(page_text, chunk_size=chunk_size, overlap=overlap), start=1):
                    if not c:
                        continue
                    chunks.append(Chunk(id=str(uuid.uuid4()), text=c, source=name, page=page_i, order=order))
        else:
            txt = simple_txt_read(io.BytesIO(data))
            for order, c in enumerate(chunk_text(txt, chunk_size=chunk_size, overlap=overlap), start=1):
                if not c:
                    continue
                chunks.append(Chunk(id=str(uuid.uuid4()), text=c, source=name, page=1, order=order))
    return chunks

def build_faiss_index(client: OpenAI, chunks: List[Chunk], model: str = EMBED_MODEL) -> IndexBundle:
    texts = [c.text for c in chunks]
    vecs = embed_texts(client, texts, model=model)
    if vecs.size == 0:
        raise ValueError("No embeddings generated — your documents may be empty.")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return IndexBundle(index=index, dim=dim, chunks=chunks, model=model)

def search(bundle: IndexBundle, client: OpenAI, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
    q = embed_texts(client, [query], model=bundle.model)
    faiss.normalize_L2(q)
    D, I = bundle.index.search(q, min(k, len(bundle.chunks)))
    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        hits.append((bundle.chunks[idx], float(score)))
    return hits

def make_prompt(query: str, contexts: List[Chunk]) -> str:
    header = (
        "You are a helpful assistant using retrieval-augmented generation.\n"
        "Answer the user's question using ONLY the provided context chunks.\n"
        "If the answer isn't in the context, say you don't know.\n"
        "Cite sources like [source.pdf p.12] inline where relevant.\n"
        "Keep the answer clear and concise."
    )
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        tag = f"[{c.source} p.{c.page}]"
        ctx_lines.append(f"### Chunk {i} {tag}\n{c.text}")
    ctx_block = "\n\n".join(ctx_lines)
    return f"{header}\n\n# CONTEXT\n{ctx_block}\n\n# QUESTION\n{query}\n\n# ANSWER"

def call_llm(client: OpenAI, prompt: str, model: str = GEN_MODEL) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="RAG AI (OpenAI + Streamlit)", page_icon="🧠", layout="wide")
st.title("🧠 RAG AI Agent")
st.caption("Upload PDFs or text files, build a vector index with OpenAI embeddings, and chat with citations. API key is read from `.env`.")

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Top-K Chunks", 3, 10, 5)
    chunk_size = st.number_input("Chunk Size (chars)", 300, 3000, 900, step=50)
    overlap = st.number_input("Chunk Overlap (chars)", 0, 800, 150, step=10)

    st.markdown("---")
    st.subheader("Index")
    uploaded_files = st.file_uploader(
        "Upload PDFs or .txt / .md files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    build_btn = st.button("🔨 Build / Rebuild Index")

if "bundle" not in st.session_state:
    st.session_state.bundle = None

client = get_client()

# Build index
if build_btn:
    if not client:
        st.error("No OpenAI API key found. Put it in `.env` as OPENAI_API_KEY.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one file.")
        st.stop()

    with st.spinner("Reading & chunking documents..."):
        files = [{"name": f.name, "data": f.getvalue()} for f in uploaded_files]
        chunks = build_corpus(files, chunk_size=int(chunk_size), overlap=int(overlap))
    st.write(f"Generated **{len(chunks)}** chunks.")

    with st.spinner("Embedding & indexing..."):
        bundle = build_faiss_index(client, chunks, model=EMBED_MODEL)
        st.session_state.bundle = bundle
    st.success("Index built and ready!")

# Chat UI
st.markdown("---")
st.subheader("Chat")

if "chat" not in st.session_state:
    st.session_state.chat = []

def ui_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content)

for m in st.session_state.chat:
    ui_message(m["role"], m["content"])

user_msg = st.chat_input("Ask a question about your documents...")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    ui_message("user", user_msg)

    if not client:
        ui_message("assistant", "Please set your OpenAI API key in `.env`.")
    elif st.session_state.bundle is None:
        ui_message("assistant", "Please build an index first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                hits = search(st.session_state.bundle, client, user_msg, k=top_k)
                if not hits:
                    st.markdown("I couldn't retrieve any relevant context.")
                contexts = [h[0] for h in hits]

                with st.expander("🔎 Retrieved context"):
                    for c, score in hits:
                        st.markdown(f"**{c.source} p.{c.page}** — sim: `{score:.3f}`")
                        st.write(c.text[:800] + ("..." if len(c.text) > 800 else ""))
                        st.markdown("---")

                prompt = make_prompt(user_msg, contexts)
                answer = call_llm(client, prompt, model=GEN_MODEL)

                st.markdown(answer)

                if contexts:
                    st.caption("Sources: " + " • ".join(f"{c.source} p.{c.page}" for c in contexts))

            st.session_state.chat.append({"role": "assistant", "content": answer})
