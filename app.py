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

# ── Optional OCR ──
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── Page config ──
st.set_page_config(page_title="FusionRAG · Doc Chat", layout="wide", page_icon="⚡")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #07060f;
    color: #e8e6f0;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(120,40,200,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(60,20,140,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 60% 20%, rgba(200,80,255,0.07) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}
h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff 30%, #b57bee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: -1px;
    margin-bottom: 0.2rem !important;
    padding-top: 1.5rem;
}
.subtitle {
    text-align: center;
    color: #7c6fa0;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    letter-spacing: 0.04em;
}
section[data-testid="stSidebar"] {
    background: #0e0b1a !important;
    border-right: 1px solid #1e1530;
}
section[data-testid="stSidebar"] * { color: #c8bfe0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed #3d2b6b !important;
    border-radius: 14px !important;
    background: rgba(61,43,107,0.12) !important;
}
.stTextInput input {
    background: #110e1f !important;
    color: #e8e6f0 !important;
    border: 1.5px solid #2d2050 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.stTextInput input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
}
.stTextInput label { color: #9d86c8 !important; font-size: 0.85rem !important; }
.stSlider label { color: #9d86c8 !important; }

/* ── AI Answer box ── */
.ai-answer {
    background: linear-gradient(145deg, #0f0b22, #16102e);
    border: 1px solid #4c1d95;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin: 1.2rem 0 0.5rem 0;
    position: relative;
    overflow: hidden;
    animation: cardIn 0.5s ease;
}
.ai-answer::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6d28d9, #a855f7, #ec4899, #a855f7, #6d28d9);
    background-size: 300% 100%;
    animation: shimmer 4s linear infinite;
}
.ai-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a855f7;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ai-body {
    font-size: 1.05rem;
    line-height: 1.9;
    color: #e2daf5;
}

/* ── Source chunks ── */
.chunks-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4b3880;
    margin: 1.8rem 0 0.6rem 0;
}
.answer-card {
    background: linear-gradient(145deg, #0e0b1c, #130f24);
    border: 1px solid #1e1840;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin: 0.5rem 0;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: cardIn 0.4s ease backwards;
}
.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1.5px;
    background: linear-gradient(90deg, #3b1f6b, #6d28d9, #3b1f6b);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
.answer-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(109,40,217,0.15);
}
.card-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6d28d9;
    margin-bottom: 0.5rem;
}
.card-body {
    font-size: 0.88rem;
    line-height: 1.75;
    color: #9d93be;
}
.hl {
    background: linear-gradient(90deg, rgba(124,58,237,0.35), rgba(168,85,247,0.25));
    border-radius: 4px;
    padding: 1px 4px;
    color: #e0d4ff;
    font-weight: 500;
}
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #3d3060;
}
.empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
hr { border-color: #1e1530 !important; margin: 1.5rem 0 !important; }

@keyframes cardIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("<h1>⚡ FusionRAG</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload documents · Ask questions · Get AI-powered answers</p>', unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 📂 Documents")
    accepted = ["pdf", "docx"]
    if OCR_AVAILABLE:
        accepted += ["png", "jpg", "jpeg"]

    files = st.file_uploader(
        "Upload PDFs or DOCX files",
        type=accepted,
        accept_multiple_files=True
    )
    st.divider()
    st.markdown("### 🤖 AI Settings")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    model_choice = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])
    st.divider()
    st.markdown("### ⚙️ Retrieval")
    top_k = st.slider("Source chunks", 1, 8, 3)
    st.caption("📌 [Get free Groq API key →](https://console.groq.com)")

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
    """Send retrieved chunks to Groq LLM and stream back the answer."""
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
                    "Answer the user's question using ONLY the provided context. "
                    "Be clear, concise, and well-structured. "
                    "If the answer isn't in the context, say so honestly."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        yield delta

# ── Session state ──
for key, default in [("db", None), ("last_file_names", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Detect file changes → reset index ──
current_names = sorted([f.name for f in files]) if files else []
if current_names != st.session_state.last_file_names:
    st.session_state.db = None
    st.session_state.last_file_names = current_names

# ── Build index ──
if files and st.session_state.db is None:
    progress = st.progress(0, text="Reading files…")
    raw_docs = []

    for i, file in enumerate(files):
        ext = file.name.rsplit(".", 1)[-1].lower()
        progress.progress(int((i / len(files)) * 40), text=f"Reading {file.name}…")
        try:
            if ext == "pdf":
                raw_docs.extend(extract_pdf(file))
            elif ext == "docx":
                raw_docs.extend(extract_docx(file))
            elif ext in ["png", "jpg", "jpeg"] and OCR_AVAILABLE:
                raw_docs.extend(extract_image(file))
        except Exception as e:
            st.warning(f"Could not process **{file.name}**: {e}")

    if not raw_docs:
        progress.empty()
        st.error("No readable text found in the uploaded files.")
        st.stop()

    progress.progress(50, text="Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "]
    )
    chunks = splitter.split_documents(raw_docs)

    progress.progress(70, text="Loading embedding model…")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    progress.progress(85, text="Building search index…")
    st.session_state.db = FAISS.from_documents(chunks, embeddings)

    progress.progress(100, text="Done!")
    progress.empty()
    st.success(f"✅ Ready! Indexed {len(chunks)} chunks from {len(files)} file(s).")

# ── Main area ──
if not files:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📄</div>
        <p>Upload PDF or DOCX files from the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    query = st.text_input(
        "Ask a question from your documents",
        placeholder="e.g. What do you understand about the Poona Pact?"
    )

    if query and st.session_state.db:
        results = st.session_state.db.similarity_search(query, k=top_k)
        st.markdown("<hr>", unsafe_allow_html=True)

        if not results:
            st.warning("No relevant content found. Try rephrasing your question.")
        else:
            # ── LLM Answer ──
            if groq_key:
                st.markdown("""
                <div class="ai-answer">
                    <div class="ai-label">✦ AI Answer</div>
                    <div class="ai-body">
                """, unsafe_allow_html=True)

                answer_placeholder = st.empty()
                full_answer = ""
                try:
                    for token in generate_answer(query, results, groq_key, model_choice):
                        full_answer += token
                        answer_placeholder.markdown(full_answer + "▌")
                    answer_placeholder.markdown(full_answer)
                except Exception as e:
                    answer_placeholder.error(f"LLM error: {e}")

                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.info("💡 Add your Groq API key in the sidebar to get AI-generated answers. It's free!", icon="🔑")

            # ── Source chunks ──
            st.markdown('<div class="chunks-header">📎 Source Chunks</div>', unsafe_allow_html=True)

            for i, res in enumerate(results, 1):
                sentences = re.split(r'(?<=[.!?])\s+', res.page_content.strip())
                snippet = " ".join(sentences[:5])
                snippet = highlight_query(snippet, query)

                src = res.metadata.get("source", "")
                page = res.metadata.get("page", "")
                src_info = f"{src} · p.{page}" if page else src
                src_tag = f'<span style="float:right;color:#3b2d6b;font-size:0.72rem;">{src_info}</span>' if src_info else ""

                st.markdown(f"""
                <div class="answer-card" style="animation-delay:{(i-1)*0.06}s">
                    <div class="card-label">Chunk {i} {src_tag}</div>
                    <div class="card-body">{snippet}</div>
                </div>
                """, unsafe_allow_html=True)
