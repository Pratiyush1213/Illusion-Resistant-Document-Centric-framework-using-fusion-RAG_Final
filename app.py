import streamlit as st
import re
import io
from groq import Groq

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from docx import Document
from pypdf import PdfReader

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

st.set_page_config(page_title="FusionRAG", layout="wide", page_icon="⚡")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Cabinet+Grotesk:wght@400;500;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #020408;
    --bg2:       #060d14;
    --bg3:       #0a1520;
    --border:    #0e2030;
    --border2:   #1a3a55;
    --accent:    #00d4ff;
    --accent2:   #0099cc;
    --accent3:   #00ffcc;
    --text:      #c8dae8;
    --text2:     #5a8099;
    --text3:     #2a4a5e;
    --glow:      rgba(0,212,255,0.12);
    --glow2:     rgba(0,255,204,0.08);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}

/* === NOISE OVERLAY === */
.stApp {
    background: var(--bg) !important;
    min-height: 100vh;
}
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.4;
}

/* === GRID BACKGROUND === */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* === HERO GLOW === */
.hero-glow {
    position: fixed;
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 500px;
    background: radial-gradient(ellipse, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}
section[data-testid="stSidebar"] * { color: var(--text2) !important; }
section[data-testid="stSidebar"] h3 {
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* sidebar badge */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 1.2rem 1rem 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.sidebar-brand-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--accent), var(--accent3));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.sidebar-brand-text {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: white !important;
    letter-spacing: -0.02em;
}
.sidebar-brand-sub {
    font-size: 0.65rem;
    color: var(--text3) !important;
    letter-spacing: 0.05em;
}

/* === FILE UPLOADER === */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border2) !important;
    border-radius: 10px !important;
    background: rgba(0,212,255,0.02) !important;
    transition: border-color 0.3s, background 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: rgba(0,212,255,0.05) !important;
}

/* === INPUTS === */
.stTextInput input, .stSelectbox select {
    background: var(--bg3) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    padding: 0.65rem 1rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.1) !important;
    outline: none !important;
}
.stTextInput label, .stSelectbox label, .stSlider label {
    color: var(--text2) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* === SLIDER === */
.stSlider > div > div > div {
    background: var(--accent) !important;
}

/* === MAIN HEADER === */
.main-header {
    text-align: center;
    padding: 3rem 0 2rem 0;
    position: relative;
}
.main-logo {
    display: inline-flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 1rem;
}
.logo-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent3) 100%);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 30px rgba(0,212,255,0.3), 0 0 60px rgba(0,212,255,0.1);
    animation: iconPulse 3s ease-in-out infinite;
}
@keyframes iconPulse {
    0%, 100% { box-shadow: 0 0 30px rgba(0,212,255,0.3), 0 0 60px rgba(0,212,255,0.1); }
    50%       { box-shadow: 0 0 40px rgba(0,212,255,0.5), 0 0 80px rgba(0,212,255,0.2); }
}
.logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent) 60%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.main-subtitle {
    color: var(--text2);
    font-size: 0.9rem;
    letter-spacing: 0.06em;
    font-weight: 400;
}
.main-subtitle span {
    color: var(--accent);
    font-weight: 600;
}

/* === STATUS CHIPS === */
.status-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    border: 1px solid;
}
.chip-green {
    background: rgba(0,255,136,0.06);
    border-color: rgba(0,255,136,0.25);
    color: #00ff88;
}
.chip-blue {
    background: rgba(0,212,255,0.06);
    border-color: rgba(0,212,255,0.25);
    color: var(--accent);
}
.chip-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: blink 2s ease infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* === QUERY BOX === */
.query-wrapper {
    max-width: 760px;
    margin: 0 auto 2rem auto;
}

/* === AI ANSWER === */
.ai-answer-wrap {
    background: linear-gradient(160deg, #040d18 0%, #060f1c 100%);
    border: 1px solid var(--border2);
    border-radius: 16px;
    padding: 0;
    margin-bottom: 1.5rem;
    overflow: hidden;
    animation: slideUp 0.4s ease;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.05), 0 20px 60px rgba(0,0,0,0.4);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ai-answer-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    background: rgba(0,212,255,0.03);
}
.ai-answer-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
}
.ai-badge-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: blink 1.5s ease infinite;
}
.ai-model-tag {
    margin-left: auto;
    font-size: 0.65rem;
    color: var(--text3);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}
.ai-answer-body {
    padding: 1.5rem;
    font-size: 1rem;
    line-height: 1.9;
    color: #daeaf5;
    font-weight: 400;
}

/* === DIVIDER === */
.section-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.5rem 0 1rem 0;
}
.section-divider-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-divider-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text3);
    font-family: 'DM Mono', monospace;
    white-space: nowrap;
}

/* === CHUNK CARDS === */
.chunk-grid {
    display: grid;
    gap: 10px;
}
.chunk-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, background 0.2s, transform 0.2s;
    animation: slideUp 0.35s ease backwards;
    cursor: default;
}
.chunk-card:hover {
    border-color: var(--border2);
    background: var(--bg3);
    transform: translateX(4px);
}
.chunk-card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
}
.chunk-num {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent2);
    font-family: 'DM Mono', monospace;
}
.chunk-source {
    font-size: 0.62rem;
    color: var(--text3);
    font-family: 'DM Mono', monospace;
    background: rgba(0,212,255,0.05);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 7px;
}
.chunk-body {
    font-size: 0.875rem;
    line-height: 1.75;
    color: var(--text2);
}
.chunk-accent-bar {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 2px;
    background: linear-gradient(180deg, var(--accent) 0%, transparent 100%);
    opacity: 0;
    transition: opacity 0.2s;
}
.chunk-card:hover .chunk-accent-bar { opacity: 1; }

/* === HIGHLIGHT === */
.hl {
    background: rgba(0,212,255,0.15);
    border-radius: 3px;
    padding: 1px 4px;
    color: var(--accent);
    font-weight: 500;
}

/* === EMPTY STATE === */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
    gap: 1rem;
}
.empty-icon {
    width: 72px; height: 72px;
    border: 1px dashed var(--border2);
    border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
    color: var(--text3);
    margin-bottom: 0.5rem;
}
.empty-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text2);
}
.empty-sub {
    font-size: 0.85rem;
    color: var(--text3);
    max-width: 320px;
    line-height: 1.6;
}

/* === PROGRESS === */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent3)) !important;
    border-radius: 100px !important;
}
.stProgress > div {
    background: var(--border) !important;
    border-radius: 100px !important;
}

/* === SUCCESS / INFO / WARNING === */
.stSuccess { border-radius: 10px !important; }
.stInfo    { border-radius: 10px !important; }
.stWarning { border-radius: 10px !important; }

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent2); }

hr { border-color: var(--border) !important; }
</style>

<div class="hero-glow"></div>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="main-header">
    <div class="main-logo">
        <div class="logo-icon">⚡</div>
        <div class="logo-text">FusionRAG</div>
    </div>
    <div class="main-subtitle">
        Retrieval-Augmented Generation · <span>AI-Powered</span> Document Intelligence
    </div>
    <div class="status-row">
        <div class="chip chip-green"><div class="chip-dot"></div>FAISS Vector Search</div>
        <div class="chip chip-blue"><div class="chip-dot"></div>Groq LLM</div>
        <div class="chip chip-green"><div class="chip-dot"></div>FastEmbed</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">⚡</div>
        <div>
            <div class="sidebar-brand-text">FusionRAG</div>
            <div class="sidebar-brand-sub">DOC INTELLIGENCE v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Documents")
    accepted = ["pdf", "docx"]
    if OCR_AVAILABLE:
        accepted += ["png", "jpg", "jpeg"]
    files = st.file_uploader(
        "Upload PDFs or DOCX files",
        type=accepted,
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### 🤖 Model")
    model_choice = st.selectbox("LLM Model", [
        # ── Production (stable) ──
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        # ── Preview (latest) ──
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen-3-32b",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("### ⚙️ Retrieval")
    top_k = st.slider("Source chunks", 1, 8, 3)

    st.divider()
    st.markdown("""
    <div style="font-size:0.68rem; color:#2a4a5e; line-height:1.6; font-family:'DM Mono',monospace;">
    API KEY → Streamlit Secrets<br>
    GROQ_API_KEY = "gsk_..."<br><br>
    <a href="https://console.groq.com" target="_blank"
       style="color:#0099cc; text-decoration:none;">
    → Get free Groq key
    </a>
    </div>
    """, unsafe_allow_html=True)

# ── Load Groq key from secrets ──
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    groq_ready = True
except Exception:
    groq_key = None
    groq_ready = False

# ── Extractors ──
def extract_pdf(file):
    reader = PdfReader(io.BytesIO(file.read()))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(LCDocument(
                page_content=text,
                metadata={"source": file.name, "page": i + 1}
            ))
    return docs

def extract_docx(file):
    doc = Document(file)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if text.strip():
        return [LCDocument(page_content=text, metadata={"source": file.name})]
    return []

def extract_image(file):
    img = Image.open(file)
    text = pytesseract.image_to_string(img)
    if text.strip():
        return [LCDocument(page_content=text, metadata={"source": file.name})]
    return []

def highlight_query(text, query):
    for word in re.findall(r'\w+', query):
        text = re.sub(rf'(?i)({re.escape(word)})', r'<span class="hl">\1</span>', text)
    return text

def generate_answer(query, chunks, api_key, model):
    context = "\n\n---\n\n".join(
        f"[Source: {c.metadata.get('source','?')} | Page: {c.metadata.get('page','?')}]\n{c.page_content}"
        for c in chunks
    )
    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert document analyst. "
                    "Answer using ONLY the provided context. "
                    "Be clear, structured, and thorough. "
                    "If the answer isn't in the context, say so honestly."
                )
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""

# ── Session state ──
for key, default in [("db", None), ("last_file_names", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Detect file changes ──
current_names = sorted([f.name for f in files]) if files else []
if current_names != st.session_state.last_file_names:
    st.session_state.db = None
    st.session_state.last_file_names = current_names

# ── Build index ──
if files and st.session_state.db is None:
    progress = st.progress(0, text="Initializing…")
    raw_docs = []
    for i, file in enumerate(files):
        ext = file.name.rsplit(".", 1)[-1].lower()
        progress.progress(int((i / len(files)) * 40), text=f"Reading {file.name}…")
        try:
            if ext == "pdf":       raw_docs.extend(extract_pdf(file))
            elif ext == "docx":    raw_docs.extend(extract_docx(file))
            elif ext in ["png","jpg","jpeg"] and OCR_AVAILABLE:
                raw_docs.extend(extract_image(file))
        except Exception as e:
            st.warning(f"Could not process **{file.name}**: {e}")

    if not raw_docs:
        progress.empty()
        st.error("No readable text found.")
        st.stop()

    progress.progress(50, text="Chunking documents…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "]
    )
    chunks = splitter.split_documents(raw_docs)

    progress.progress(70, text="Loading embedding model…")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    progress.progress(88, text="Building FAISS index…")
    st.session_state.db = FAISS.from_documents(chunks, embeddings)

    progress.progress(100, text="Ready!")
    progress.empty()
    st.success(f"✅ Indexed {len(chunks)} chunks from {len(files)} file(s).")

# ── Main ──
if not files:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📄</div>
        <div class="empty-title">No documents loaded</div>
        <div class="empty-sub">Upload PDF or DOCX files from the sidebar to start querying your documents.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="query-wrapper">', unsafe_allow_html=True)
    query = st.text_input(
        "query",
        placeholder="Ask anything from your documents…",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if query and st.session_state.db:
        results = st.session_state.db.similarity_search(query, k=top_k)

        if not results:
            st.warning("No relevant content found. Try rephrasing.")
        else:
            # ── AI Answer ──
            if groq_ready:
                model_short = model_choice.split("-")[0].upper() + " · " + model_choice
                st.markdown(f"""
                <div class="ai-answer-wrap">
                    <div class="ai-answer-header">
                        <div class="ai-answer-badge">
                            <div class="ai-badge-dot"></div>
                            AI Answer
                        </div>
                        <div class="ai-model-tag">{model_choice}</div>
                    </div>
                    <div class="ai-answer-body" id="ai-body">
                """, unsafe_allow_html=True)

                placeholder = st.empty()
                full = ""
                try:
                    for token in generate_answer(query, results, groq_key, model_choice):
                        full += token
                        placeholder.markdown(
                            f'<div style="font-size:1rem;line-height:1.9;color:#daeaf5;padding:0 1.5rem 1.5rem 1.5rem">{full}▌</div>',
                            unsafe_allow_html=True
                        )
                    placeholder.markdown(
                        f'<div style="font-size:1rem;line-height:1.9;color:#daeaf5;padding:0 1.5rem 1.5rem 1.5rem">{full}</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    placeholder.error(f"LLM error: {e}")

                st.markdown('</div></div>', unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.15);
                            border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.2rem;
                            font-size:0.88rem;color:#5a8099;">
                    🔑 Add <code style="color:#00d4ff">GROQ_API_KEY</code> to Streamlit Secrets to enable AI answers.
                </div>
                """, unsafe_allow_html=True)

            # ── Source Chunks ──
            st.markdown("""
            <div class="section-divider">
                <div class="section-divider-line"></div>
                <div class="section-divider-label">Source Chunks</div>
                <div class="section-divider-line"></div>
            </div>
            <div class="chunk-grid">
            """, unsafe_allow_html=True)

            for i, res in enumerate(results, 1):
                sentences = re.split(r'(?<=[.!?])\s+', res.page_content.strip())
                snippet = " ".join(sentences[:5])
                snippet = highlight_query(snippet, query)
                src = res.metadata.get("source", "")
                page = res.metadata.get("page", "")
                src_info = f"{src} · p.{page}" if page else src

                st.markdown(f"""
                <div class="chunk-card" style="animation-delay:{(i-1)*0.07}s">
                    <div class="chunk-accent-bar"></div>
                    <div class="chunk-card-top">
                        <div class="chunk-num">Chunk {i:02d}</div>
                        <div class="chunk-source">{src_info}</div>
                    </div>
                    <div class="chunk-body">{snippet}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
