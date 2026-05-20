import streamlit as st
import time
import os
import hashlib
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

DEFAULT_OLLAMA_MODEL = "SpeakLeash/bielik-7b-instruct-v0.1-gguf"
OLLAMA_MODEL = os.getenv("MODEL_NAME", DEFAULT_OLLAMA_MODEL)
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
MEMORY_RESULTS = int(os.getenv("MEMORY_RESULTS", "4"))
RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "6"))
DOCS_DIR = os.getenv("DOCS_DIR")
DOCS_FILE = os.getenv("DOCS_FILE")
DEFAULT_RAG_DIR_NAME = ".OAP_RAG"
MEMORY_DIR = os.getenv(
    "MEMORY_DIR",
    str(Path(DOCS_DIR).expanduser() / DEFAULT_RAG_DIR_NAME)
    if DOCS_DIR
    else str(Path(DOCS_FILE).expanduser().parent / DEFAULT_RAG_DIR_NAME)
    if DOCS_FILE
    else f"./{DEFAULT_RAG_DIR_NAME}",
)
DOCS_RESULTS = int(os.getenv("DOCS_RESULTS", "6"))
DOCS_CHUNK_SIZE = int(os.getenv("DOCS_CHUNK_SIZE", "1500"))
DOCS_CHUNK_OVERLAP = int(os.getenv("DOCS_CHUNK_OVERLAP", "200"))
DOCS_EXTENSIONS = {
    extension.strip().lower()
    for extension in os.getenv(
        "DOCS_EXTENSIONS",
        ".txt,.md,.py,.json,.yaml,.yml,.toml,.ini,.cfg,.pdf",
    ).split(",")
    if extension.strip()
}
PROMPT_TEMPLATE = """
Jestes pomocnym asystentem.

Relevant long-term memory:
{memory_context}

Relevant project files:
{document_context}

Recent conversation:
{recent_conversation}

Current user question:
{question}

Odpowiedz na pytanie, korzystajac z pamieci i plikow tylko wtedy, gdy sa istotne.
"""
CHAT_TITLE = "ollama-app-plus"
CHAT_HINT = "Co tam?"
DEFAULT_TOKEN_RATE = 20.0
MIN_TOKEN_DELAY_SECONDS = 0.005
MAX_TOKEN_DELAY_SECONDS = 0.2


@dataclass
class TokenRateEstimator:
    """Tracks an approximate token rate with EWMA smoothing."""

    tokens_per_second: float = DEFAULT_TOKEN_RATE
    smoothing: float = 0.35

    def update(self, token_count: int, elapsed_seconds: float) -> None:
        if token_count <= 0 or elapsed_seconds <= 0:
            return

        sample_rate = token_count / elapsed_seconds
        self.tokens_per_second = (
            self.smoothing * sample_rate
            + (1 - self.smoothing) * self.tokens_per_second
        )

    @property
    def delay_seconds(self) -> float:
        if self.tokens_per_second <= 0:
            return 1 / DEFAULT_TOKEN_RATE

        return min(
            MAX_TOKEN_DELAY_SECONDS,
            max(MIN_TOKEN_DELAY_SECONDS, 1 / self.tokens_per_second),
        )


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def get_token_rate_estimator() -> TokenRateEstimator:
    if "token_rate_estimator" not in st.session_state:
        st.session_state.token_rate_estimator = TokenRateEstimator()

    return st.session_state.token_rate_estimator


def get_chat_history() -> list[dict[str, str]]:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    return st.session_state.chat_history


def get_memory_store():
    return get_chroma_store(
        session_key="memory_store",
        collection_name="conversation_memory",
        status_key="memory_status",
        status_label="Long-term memory",
    )


def get_document_store():
    has_uploaded_document = bool(st.session_state.get("uploaded_document_source"))
    if not DOCS_DIR and not DOCS_FILE and not has_uploaded_document:
        st.session_state.document_status = (
            "Document memory disabled: choose a file or set DOCS_DIR/DOCS_FILE."
        )
        return None

    return get_chroma_store(
        session_key="document_store",
        collection_name="project_documents",
        status_key="document_status",
        status_label="Document memory",
    )


def get_chroma_store(
    session_key: str,
    collection_name: str,
    status_key: str,
    status_label: str,
):
    if Chroma is None:
        st.session_state[status_key] = (
            f"{status_label} disabled: install langchain-chroma."
        )
        return None

    if session_key not in st.session_state:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        st.session_state[session_key] = Chroma(
            collection_name=collection_name,
            persist_directory=MEMORY_DIR,
            embedding_function=embeddings,
        )

    st.session_state[status_key] = (
        f"{status_label}: {MEMORY_DIR} via {EMBEDDING_MODEL}"
    )
    return st.session_state[session_key]


def retrieve_memory_context(user_input: str) -> str:
    memory_store = get_memory_store()
    if memory_store is None:
        return "No long-term memory available."

    try:
        docs = memory_store.similarity_search(user_input, k=MEMORY_RESULTS)
    except Exception as exc:
        st.session_state.memory_status = f"Long-term memory unavailable: {exc}"
        return "No long-term memory available."

    if not docs:
        return "No relevant long-term memory found."

    return "\n".join(f"- {doc.page_content}" for doc in docs)


def retrieve_document_context(user_input: str) -> str:
    document_store = get_document_store()
    if document_store is None:
        return "No project files available."

    try:
        docs = document_store.similarity_search(user_input, k=DOCS_RESULTS)
    except Exception as exc:
        st.session_state.document_status = f"Document memory unavailable: {exc}"
        return "No project files available."

    if not docs:
        return "No relevant project files found."

    formatted_docs = []
    for doc in docs:
        source_path = doc.metadata.get("source_path", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        formatted_docs.append(
            f"Source: {source_path} chunk {chunk_index}\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


def format_recent_conversation() -> str:
    history = get_chat_history()[-RECENT_MESSAGES_LIMIT:]
    if not history:
        return "No recent conversation yet."

    return "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in history
    )


def remember_exchange(user_input: str, assistant_response: str) -> None:
    memory_store = get_memory_store()
    if memory_store is None:
        return

    memory_text = f"User: {user_input}\nAssistant: {assistant_response}"
    metadata = {
        "model": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        memory_store.add_texts(
            texts=[memory_text],
            metadatas=[metadata],
            ids=[str(uuid4())],
        )
    except Exception as exc:
        st.session_state.memory_status = f"Could not save long-term memory: {exc}"


def iter_document_files(docs_dir: Path):
    ignored_dirs = {".git", ".venv", "__pycache__", "memory_db", DEFAULT_RAG_DIR_NAME}
    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in DOCS_EXTENSIONS:
            continue
        yield path


def iter_document_sources():
    if DOCS_FILE:
        path = Path(DOCS_FILE).expanduser().resolve()
        if path.exists() and path.is_file():
            yield path, path.name
        else:
            st.session_state.document_status = f"Document file not found: {path}"
        return

    if not DOCS_DIR:
        return

    docs_dir = Path(DOCS_DIR).expanduser().resolve()
    if not docs_dir.exists() or not docs_dir.is_dir():
        st.session_state.document_status = f"Document directory not found: {docs_dir}"
        return

    for path in iter_document_files(docs_dir):
        yield path, str(path.relative_to(docs_dir))


def read_pdf_text(path: Path) -> str:
    if PdfReader is None:
        st.session_state.document_status = "PDF support disabled: install pypdf."
        return ""

    reader = PdfReader(str(path))
    return extract_pdf_text(reader)


def read_pdf_bytes(content: bytes) -> str:
    if PdfReader is None:
        st.session_state.document_status = "PDF support disabled: install pypdf."
        return ""

    reader = PdfReader(BytesIO(content))
    return extract_pdf_text(reader)


def extract_pdf_text(reader) -> str:
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text}")

    return "\n\n".join(pages)


def read_document_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf_text(path)

    return path.read_text(encoding="utf-8", errors="ignore")


def read_uploaded_document_text(file_name: str, content: bytes) -> str:
    if Path(file_name).suffix.lower() == ".pdf":
        return read_pdf_bytes(content)

    return content.decode("utf-8", errors="ignore")


def chunk_text(text: str) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0
    step = max(1, DOCS_CHUNK_SIZE - DOCS_CHUNK_OVERLAP)

    while start < len(text):
        chunk = text[start : start + DOCS_CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def document_id(relative_path: str, file_hash: str, chunk_index: int) -> str:
    raw_id = f"{relative_path}:{file_hash}:{chunk_index}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def clear_document_chunks(document_store, relative_path: str) -> None:
    collection = getattr(document_store, "_collection", None)
    if collection is None:
        return

    collection.delete(where={"source_path": relative_path})


def build_document_chunks(
    relative_path: str,
    content: str,
) -> tuple[list[str], list[dict[str, str | int]], list[str]]:
    file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    chunks = chunk_text(content)
    texts = []
    metadatas = []
    ids = []

    for chunk_index, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append(
            {
                "source": "project_documents",
                "source_path": relative_path,
                "chunk_index": chunk_index,
                "file_hash": file_hash,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        ids.append(document_id(relative_path, file_hash, chunk_index))

    return texts, metadatas, ids


def add_document_to_store(
    document_store,
    relative_path: str,
    content: str,
) -> int:
    chunks, metadatas, ids = build_document_chunks(relative_path, content)
    if not chunks:
        return 0

    try:
        clear_document_chunks(document_store, relative_path)
    except Exception:
        pass

    document_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def index_uploaded_document(file_name: str, content: bytes) -> tuple[int, int]:
    document_store = get_chroma_store(
        session_key="document_store",
        collection_name="project_documents",
        status_key="document_status",
        status_label="Document memory",
    )
    if document_store is None:
        return 0, 0

    suffix = Path(file_name).suffix.lower()
    if suffix not in DOCS_EXTENSIONS:
        st.session_state.document_status = f"Unsupported document type: {suffix}"
        return 0, 0

    text = read_uploaded_document_text(file_name, content)
    chunk_count = add_document_to_store(document_store, file_name, text)
    indexed_files = 1 if chunk_count else 0
    st.session_state.uploaded_document_source = file_name
    st.session_state.document_status = (
        f"Indexed {indexed_files} uploaded files and {chunk_count} chunks from {file_name}"
    )
    return indexed_files, chunk_count


def index_documents() -> tuple[int, int]:
    document_store = get_document_store()
    if document_store is None:
        return 0, 0

    source_label = DOCS_FILE or DOCS_DIR or "document source"
    indexed_files = 0
    indexed_chunks = 0

    for path, relative_path in iter_document_sources():
        if path.suffix.lower() not in DOCS_EXTENSIONS:
            continue

        try:
            content = read_document_text(path)
        except OSError:
            continue

        chunk_count = add_document_to_store(document_store, relative_path, content)
        if chunk_count:
            indexed_files += 1
            indexed_chunks += chunk_count

    st.session_state.document_status = (
        f"Indexed {indexed_files} files and {indexed_chunks} chunks from {source_label}"
    )
    return indexed_files, indexed_chunks


def response_generator(user_input):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    model = OllamaLLM(model=OLLAMA_MODEL)

    chain = prompt | model | StrOutputParser()
    estimator = get_token_rate_estimator()
    chain_input = {
        "memory_context": retrieve_memory_context(user_input),
        "document_context": retrieve_document_context(user_input),
        "recent_conversation": format_recent_conversation(),
        "question": user_input,
    }

    previous_chunk_at = time.perf_counter()
    streamed_chunks = 0

    try:
        for chunk in chain.stream(chain_input):
            if not chunk:
                continue

            now = time.perf_counter()
            estimator.update(estimate_token_count(chunk), now - previous_chunk_at)
            previous_chunk_at = now
            streamed_chunks += 1

            yield chunk
    except NotImplementedError:
        streamed_chunks = 0

    if streamed_chunks > 0:
        return

    response_started_at = time.perf_counter()
    response = chain.invoke(chain_input)
    response_elapsed = time.perf_counter() - response_started_at
    estimator.update(estimate_token_count(response), response_elapsed)

    for word in response.split():
        yield word + " "
        time.sleep(estimator.delay_seconds)


st.title(CHAT_TITLE)
st.caption(f"Model: {OLLAMA_MODEL}")

estimator = get_token_rate_estimator()
st.caption(
    f"Estimated token rate: {estimator.tokens_per_second:.1f} tokens/s "
    f"(fallback delay: {estimator.delay_seconds:.3f}s)"
)
st.caption(st.session_state.get("memory_status", "Long-term memory initializing."))
st.caption(st.session_state.get("document_status", "Document memory initializing."))

with st.sidebar:
    st.subheader("Document memory")
    uploaded_document = st.file_uploader(
        "Choose one document",
        type=sorted(extension.lstrip(".") for extension in DOCS_EXTENSIONS),
    )
    if uploaded_document is not None:
        if st.button("Index selected file"):
            with st.spinner("Indexing selected file..."):
                index_uploaded_document(
                    uploaded_document.name,
                    uploaded_document.getvalue(),
                )
            st.rerun()

    st.caption(f"DOCS_DIR: {DOCS_DIR or 'not set'}")
    st.caption(f"DOCS_FILE: {DOCS_FILE or 'not set'}")
    st.caption(f"RAG dir: {MEMORY_DIR}")
    st.caption(f"Extensions: {', '.join(sorted(DOCS_EXTENSIONS))}")
    st.caption(f"Results: {DOCS_RESULTS}")
    if st.button(
        "Reindex configured source",
        disabled=not bool(DOCS_DIR or DOCS_FILE),
    ):
        with st.spinner("Indexing documents..."):
            index_documents()
        st.rerun()

for message in get_chat_history():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input(CHAT_HINT):
    chat_history = get_chat_history()
    chat_history.append({"role": "user", "content": user_input})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(user_input))

    chat_history.append({"role": "assistant", "content": response})
    remember_exchange(user_input, response)
