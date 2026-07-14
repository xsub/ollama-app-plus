import streamlit as st
import time
import os
import hashlib
import fnmatch
import logging
import re
import zipfile
from contextlib import contextmanager
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree
from uuid import uuid4
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
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
DOCS_FILE_MASK = os.getenv("DOCS_FILE_MASK", os.getenv("DOCS_GLOB", "**/*"))
SUPPRESS_PDF_WARNINGS = os.getenv("SUPPRESS_PDF_WARNINGS", "1").lower() not in {
    "0",
    "false",
    "no",
}
DOCS_EXTENSIONS = {
    extension.strip().lower()
    for extension in os.getenv(
        "DOCS_EXTENSIONS",
        ".txt,.md,.py,.json,.yaml,.yml,.toml,.ini,.cfg,.pdf,.docx",
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
AGENT_SYSTEM_TEMPLATE = """
Jestes pomocnym asystentem.

Masz dostep do narzedzi search_local_documents oraz get_document_index_status.
Uzyj search_local_documents, gdy pytanie wymaga informacji z lokalnych plikow
projektu, raportow, dokumentow PDF albo DOCX.
Uzyj get_document_index_status, gdy uzytkownik pyta ile plikow jest
zindeksowanych, ktore pliki sa zindeksowane, albo jaki jest status/progress
indeksowania.
Jesli uzytkownik poda maske plikow, przekaz ja jako argument file_mask narzedzia.
Maska ustawiona w sidebarze jest nadrzednym ograniczeniem i nie wolno jej
rozszerzac szersza maska z narzedzia.
Nazwy plikow sa czescia wiedzy: moga zawierac firme, stanowisko i date aplikacji.
Nie zgaduj danych z dokumentow. Jesli narzedzie nie znajdzie informacji,
powiedz to jasno.
Nie opisuj procesu ani nazw narzedzi, chyba ze uzytkownik wprost o to zapyta.

Relevant long-term memory:
{memory_context}

Recent conversation:
{recent_conversation}

Odpowiedz na pytanie uzytkownika po polsku, chyba ze uzytkownik poprosi o inny jezyk.
"""
CHAT_TITLE = "streamlit-langchain-ai-agent-rag-experiment"
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
    has_uploaded_document = bool(
        st.session_state.get("uploaded_document_sources")
        or st.session_state.get("uploaded_document_source")
    )
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


def get_uploaded_document_sources() -> list[str]:
    sources = list(st.session_state.get("uploaded_document_sources", []))
    legacy_source = st.session_state.get("uploaded_document_source")

    if legacy_source:
        sources.extend(
            source.strip() for source in str(legacy_source).split(",") if source.strip()
        )

    return sorted(set(sources))


def source_paths_for_file_mask(file_mask: str | None) -> list[str]:
    masks = parse_file_masks(file_mask)
    source_paths = []

    for _, relative_path in iter_document_sources(file_mask):
        source_paths.append(relative_path)

    for uploaded_source in get_uploaded_document_sources():
        if matches_source_path(uploaded_source, masks):
            source_paths.append(uploaded_source)

    return sorted(set(source_paths))


def format_document_source_list(source_paths: list[str]) -> str:
    if not source_paths:
        return "Indexed document sources: none."

    return "Indexed document sources:\n" + "\n".join(
        f"- {source_path}" for source_path in source_paths
    )


def retrieve_document_context(user_input: str, file_mask: str | None = None) -> str:
    document_store = get_document_store()
    if document_store is None:
        return "No project files available."

    search_kwargs = {}
    source_paths = []
    if file_mask:
        source_paths = source_paths_for_file_mask(file_mask)
        if not source_paths:
            return f"No project files match mask: {file_mask}"
        search_kwargs["filter"] = {"source_path": {"$in": source_paths}}

    try:
        docs = document_store.similarity_search(
            user_input,
            k=DOCS_RESULTS,
            **search_kwargs,
        )
    except Exception as exc:
        if file_mask:
            try:
                docs = document_store.similarity_search(
                    user_input,
                    k=max(DOCS_RESULTS * 4, DOCS_RESULTS),
                )
            except Exception as fallback_exc:
                st.session_state.document_status = (
                    f"Document memory unavailable: {fallback_exc}"
                )
                return "No project files available."

            masks = parse_file_masks(file_mask)
            docs = [
                doc
                for doc in docs
                if matches_source_path(doc.metadata.get("source_path", ""), masks)
            ][:DOCS_RESULTS]
        else:
            st.session_state.document_status = f"Document memory unavailable: {exc}"
            return "No project files available."

    if not docs:
        if file_mask:
            return (
                f"{format_document_source_list(source_paths)}\n\n"
                f"No relevant project file chunks found for mask: {file_mask}"
            )
        return "No relevant project files found."

    formatted_docs = [format_document_source_list(source_paths)] if source_paths else []
    for doc in docs:
        source_path = doc.metadata.get("source_path", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        chunk_kind = doc.metadata.get("chunk_kind", "content")
        formatted_docs.append(
            f"Source: {source_path} {chunk_kind} chunk {chunk_index}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


def get_document_collection_metadatas() -> list[dict]:
    document_store = get_document_store()
    if document_store is None:
        return []

    collection = getattr(document_store, "_collection", None)
    if collection is None:
        return []

    try:
        collection_data = collection.get(include=["metadatas"])
    except Exception as exc:
        st.session_state.document_status = f"Could not read document index: {exc}"
        return []

    return [
        metadata
        for metadata in collection_data.get("metadatas", [])
        if metadata and metadata.get("source_path")
    ]


def get_document_index_snapshot(file_mask: str | None = None) -> dict:
    masks = parse_file_masks(file_mask)
    metadatas = get_document_collection_metadatas()
    source_paths = []
    indexed_at_by_source = {}

    for metadata in metadatas:
        source_path = metadata.get("source_path", "")
        if file_mask and not matches_source_path(source_path, masks):
            continue
        source_paths.append(source_path)
        indexed_at = metadata.get("indexed_at", "")
        if indexed_at and indexed_at > indexed_at_by_source.get(source_path, ""):
            indexed_at_by_source[source_path] = indexed_at

    unique_source_paths = sorted(set(source_paths))
    return {
        "file_count": len(unique_source_paths),
        "chunk_count": len(source_paths),
        "source_paths": unique_source_paths,
        "indexed_at_by_source": indexed_at_by_source,
        "file_mask": file_mask or "",
        "document_status": st.session_state.get("document_status", ""),
        "progress_status": st.session_state.get("index_progress_status", ""),
    }


def format_document_index_snapshot(snapshot: dict) -> str:
    source_paths = snapshot["source_paths"]
    preview_limit = 25
    preview_paths = source_paths[:preview_limit]
    remaining_count = max(0, len(source_paths) - preview_limit)
    lines = [
        f"Liczba zindeksowanych plikow: {snapshot['file_count']}",
        f"Liczba zindeksowanych chunkow: {snapshot['chunk_count']}",
        f"Aktywna maska plikow: {snapshot['file_mask'] or 'bez ograniczenia'}",
    ]

    if snapshot["document_status"]:
        lines.append(f"Status dokumentow: {snapshot['document_status']}")
    if snapshot["progress_status"]:
        lines.append(f"Postep indeksowania: {snapshot['progress_status']}")

    if preview_paths:
        lines.append("Zindeksowane pliki:")
        lines.extend(f"- {source_path}" for source_path in preview_paths)
        if remaining_count:
            lines.append(f"- ... i jeszcze {remaining_count}")

    return "\n".join(lines)


def is_index_status_question(user_input: str) -> bool:
    normalized = user_input.lower()
    normalized = (
        normalized.replace("ą", "a")
        .replace("ć", "c")
        .replace("ę", "e")
        .replace("ł", "l")
        .replace("ń", "n")
        .replace("ó", "o")
        .replace("ś", "s")
        .replace("ż", "z")
        .replace("ź", "z")
    )

    asks_count = ("ile" in normalized or "how many" in normalized) and (
        "plik" in normalized or "file" in normalized
    )
    mentions_index = "zindeks" in normalized or "indexed" in normalized
    asks_list = (
        "ktore pliki" in normalized
        or "jakie pliki" in normalized
        or "lista plik" in normalized
        or "list files" in normalized
    )
    asks_progress = (
        "progress" in normalized
        or "postep" in normalized
        or ("status" in normalized and "indeks" in normalized)
    )

    return (asks_count and mentions_index) or asks_list or asks_progress


def extract_date_from_source_path(source_path: str) -> tuple[str, tuple[int, int, int]]:
    match = re.search(
        r"(?P<year>20\d{2})[-_. ]?(?P<month>0[1-9]|1[0-2])?"
        r"[-_. ]?(?P<day>0[1-9]|[12]\d|3[01])?",
        source_path,
    )
    if not match:
        return "", (0, 0, 0)

    year = int(match.group("year"))
    month = int(match.group("month") or 0)
    day = int(match.group("day") or 0)
    if day:
        return f"{year:04d}-{month:02d}-{day:02d}", (year, month, day)
    if month:
        return f"{year:04d}-{month:02d}", (year, month, 0)
    return f"{year:04d}", (year, 0, 0)


def strip_application_filename_noise(title: str) -> str:
    title = re.sub(
        r"\b20\d{2}(?:[-_. ]?(?:0[1-9]|1[0-2]))?"
        r"(?:[-_. ]?(?:0[1-9]|[12]\d|3[01]))?\b",
        " ",
        title,
    )
    title = re.sub(r"\b(?:pawel|paweł|suchanecki|cv|lom|lo m)\b", " ", title, flags=re.I)
    return " ".join(title.split())


def split_company_and_role(title: str) -> tuple[str, str]:
    role_patterns = [
        "senior scrum master",
        "scrum master",
        "agile delivery lead",
        "agile delivery",
        "senior agile coach",
        "agile coach",
        "development manager",
        "delivery manager",
        "engineering manager",
        "project manager",
        "product owner",
        "product manager",
    ]
    lowered = title.lower()
    matches = [
        (lowered.find(pattern), pattern)
        for pattern in role_patterns
        if lowered.find(pattern) >= 0
    ]
    if not matches:
        return "", title

    role_start, _ = min(matches, key=lambda item: item[0])
    company = title[:role_start].strip(" -_")
    role = title[role_start:].strip(" -_")
    return company, role


def parse_application_from_source_path(source_path: str, indexed_at: str = "") -> dict:
    file_name = Path(source_path).name
    title = readable_filename(file_name)
    date_label, date_key = extract_date_from_source_path(file_name)
    clean_title = strip_application_filename_noise(title)
    company, role = split_company_and_role(clean_title)
    indexed_key = indexed_at or ""

    return {
        "source_path": source_path,
        "file_name": file_name,
        "date_label": date_label,
        "date_key": date_key,
        "indexed_at": indexed_at,
        "indexed_key": indexed_key,
        "title": clean_title or title,
        "company": company,
        "role": role or clean_title or title,
    }


def is_application_source_path(source_path: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", source_path.lower())
    return any(token in normalized.split() for token in {"cv", "cvs", "lom", "loms"})


def get_recent_application_records(file_mask: str | None = None) -> list[dict]:
    snapshot = get_document_index_snapshot(file_mask)
    indexed_at_by_source = snapshot.get("indexed_at_by_source", {})
    records = [
        parse_application_from_source_path(
            source_path,
            indexed_at_by_source.get(source_path, ""),
        )
        for source_path in snapshot["source_paths"]
        if is_application_source_path(source_path)
    ]

    return sorted(
        records,
        key=lambda record: (
            record["date_key"],
            record["indexed_key"],
            record["source_path"],
        ),
        reverse=True,
    )


def is_recent_application_question(user_input: str) -> bool:
    normalized = user_input.lower()
    normalized = (
        normalized.replace("ą", "a")
        .replace("ć", "c")
        .replace("ę", "e")
        .replace("ł", "l")
        .replace("ń", "n")
        .replace("ó", "o")
        .replace("ś", "s")
        .replace("ż", "z")
        .replace("ź", "z")
    )

    mentions_application = (
        "aplikow" in normalized
        or "wyslal cv" in normalized
        or "cv" in normalized
    )
    asks_recent = "ostatnio" in normalized or "najnowsz" in normalized
    asks_target = "gdzie" in normalized or "rola" in normalized or "role" in normalized
    return mentions_application and asks_recent and asks_target


def format_recent_applications_response(records: list[dict]) -> str:
    if not records:
        return (
            "Nie znalazlem zindeksowanych plikow CV/LoM pasujacych do aktywnej maski."
        )

    top_date_key = records[0]["date_key"]
    if top_date_key != (0, 0, 0):
        selected_records = [
            record for record in records if record["date_key"] == top_date_key
        ]
        intro = (
            f"Najnowsza data z nazw plikow to {records[0]['date_label']}. "
            "Pasujace najnowsze aplikacje:"
        )
    else:
        selected_records = records[:5]
        intro = (
            "Nie widze dat w nazwach plikow, wiec pokazuje ostatnie wpisy "
            "wedlug metadanych indeksu:"
        )

    lines = [intro]
    for record in selected_records[:10]:
        target = record["title"]
        if record["company"] and record["role"]:
            target = f"{record['company']} — {record['role']}"
        elif record["role"]:
            target = record["role"]

        date_prefix = f"{record['date_label']}: " if record["date_label"] else ""
        lines.append(f"- {date_prefix}{target} ({record['source_path']})")

    if len(selected_records) > 10:
        lines.append(f"- ... i jeszcze {len(selected_records) - 10}")

    return "\n".join(lines)


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


def parse_file_masks(mask_text: str | None) -> list[str]:
    if not mask_text:
        return ["**/*"]

    masks = []
    for line in mask_text.splitlines():
        masks.extend(part.strip() for part in line.split(",") if part.strip())

    return masks or ["**/*"]


def get_active_file_mask() -> str:
    return st.session_state.get("docs_file_mask", DOCS_FILE_MASK)


def masks_are_unrestricted(mask_text: str | None) -> bool:
    unrestricted_masks = {"*", "**", "**/*"}
    masks = [mask.replace("\\", "/") for mask in parse_file_masks(mask_text)]
    return any(mask in unrestricted_masks for mask in masks)


def get_effective_tool_file_mask(tool_file_mask: str | None = None) -> str:
    active_mask = get_active_file_mask()
    if not masks_are_unrestricted(active_mask):
        return active_mask
    if tool_file_mask and not masks_are_unrestricted(tool_file_mask):
        return tool_file_mask
    return active_mask


def matches_file_masks(path: Path, root: Path, masks: list[str]) -> bool:
    relative_path = path.relative_to(root).as_posix()
    file_name = path.name

    for mask in masks:
        normalized_mask = mask.replace("\\", "/")
        if normalized_mask in {"*", "**", "**/*"}:
            return True
        if fnmatch.fnmatchcase(relative_path, normalized_mask):
            return True
        if fnmatch.fnmatchcase(file_name, normalized_mask):
            return True
        if normalized_mask.startswith("**/") and fnmatch.fnmatchcase(
            relative_path, normalized_mask[3:]
        ):
            return True

    return False


def matches_source_path(source_path: str, masks: list[str]) -> bool:
    path = Path(source_path)
    for mask in masks:
        normalized_mask = mask.replace("\\", "/")
        if normalized_mask in {"*", "**", "**/*"}:
            return True
        if fnmatch.fnmatchcase(source_path.replace("\\", "/"), normalized_mask):
            return True
        if fnmatch.fnmatchcase(path.name, normalized_mask):
            return True
        if normalized_mask.startswith("**/") and fnmatch.fnmatchcase(
            source_path.replace("\\", "/"), normalized_mask[3:]
        ):
            return True

    return False


def iter_document_files(docs_dir: Path, file_mask: str | None = None):
    ignored_dirs = {".git", ".venv", "__pycache__", "memory_db", DEFAULT_RAG_DIR_NAME}
    masks = parse_file_masks(file_mask)

    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in DOCS_EXTENSIONS:
            continue
        if not matches_file_masks(path, docs_dir, masks):
            continue
        yield path


def iter_document_sources(file_mask: str | None = None):
    masks = parse_file_masks(file_mask)

    if DOCS_FILE:
        path = Path(DOCS_FILE).expanduser().resolve()
        if path.exists() and path.is_file():
            if matches_source_path(path.name, masks):
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

    for path in iter_document_files(docs_dir, file_mask):
        yield path, str(path.relative_to(docs_dir))


@contextmanager
def quiet_pdf_parser_warnings():
    if not SUPPRESS_PDF_WARNINGS:
        yield
        return

    logger_names = [
        "pypdf",
        "pypdf._reader",
        "pypdf._utils",
        "pypdf.generic",
        "pypdf.generic._data_structures",
    ]
    loggers = [logging.getLogger(logger_name) for logger_name in logger_names]
    previous_levels = [logger.level for logger in loggers]

    try:
        for logger in loggers:
            logger.setLevel(logging.ERROR)
        yield
    finally:
        for logger, previous_level in zip(loggers, previous_levels):
            logger.setLevel(previous_level)


def read_pdf_text(path: Path) -> str:
    if PdfReader is None:
        st.session_state.document_status = "PDF support disabled: install pypdf."
        return ""

    with quiet_pdf_parser_warnings():
        reader = PdfReader(str(path))
        return extract_pdf_text(reader)


def read_pdf_bytes(content: bytes) -> str:
    if PdfReader is None:
        st.session_state.document_status = "PDF support disabled: install pypdf."
        return ""

    with quiet_pdf_parser_warnings():
        reader = PdfReader(BytesIO(content))
        return extract_pdf_text(reader)


def extract_pdf_text(reader) -> str:
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text}")

    return "\n\n".join(pages)


def read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return extract_docx_text(archive)


def read_docx_bytes(content: bytes) -> str:
    with zipfile.ZipFile(BytesIO(content)) as archive:
        return extract_docx_text(archive)


def extract_docx_text(archive: zipfile.ZipFile) -> str:
    word_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    text_tag = f"{word_namespace}t"
    tab_tag = f"{word_namespace}tab"
    break_tag = f"{word_namespace}br"
    paragraph_tag = f"{word_namespace}p"
    xml_paths = [
        name
        for name in archive.namelist()
        if name == "word/document.xml"
        or name.startswith("word/header")
        or name.startswith("word/footer")
    ]

    paragraphs = []
    for xml_path in xml_paths:
        try:
            root = ElementTree.fromstring(archive.read(xml_path))
        except ElementTree.ParseError:
            continue

        for paragraph in root.iter(paragraph_tag):
            parts = []
            for node in paragraph.iter():
                if node.tag == text_tag and node.text:
                    parts.append(node.text)
                elif node.tag == tab_tag:
                    parts.append("\t")
                elif node.tag == break_tag:
                    parts.append("\n")

            paragraph_text = "".join(parts).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)

    return "\n\n".join(paragraphs)


def read_document_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf_text(path)
    if path.suffix.lower() == ".docx":
        return read_docx_text(path)

    return path.read_text(encoding="utf-8", errors="ignore")


def read_uploaded_document_text(file_name: str, content: bytes) -> str:
    if Path(file_name).suffix.lower() == ".pdf":
        return read_pdf_bytes(content)
    if Path(file_name).suffix.lower() == ".docx":
        return read_docx_bytes(content)

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


def readable_filename(relative_path: str) -> str:
    path = Path(relative_path)
    name_without_suffix = path.stem
    normalized = name_without_suffix.replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def build_document_metadata_text(relative_path: str) -> str:
    path = Path(relative_path)
    readable_name = readable_filename(relative_path)
    return "\n".join(
        [
            f"Document source path: {relative_path}",
            f"Document file name: {path.name}",
            f"Document title from filename: {readable_name}",
            f"Document extension: {path.suffix.lower() or 'unknown'}",
            (
                "Filename hints may include person name, company, role, job title, "
                "application target, report name, or date. Use these hints when the "
                "user asks which document was indexed or where a CV/application was sent."
            ),
        ]
    )


def clear_document_chunks(document_store, relative_path: str) -> None:
    collection = getattr(document_store, "_collection", None)
    if collection is None:
        return

    collection.delete(where={"source_path": relative_path})


def prune_document_chunks_outside_mask(document_store, file_mask: str | None) -> int:
    if not file_mask:
        return 0

    collection = getattr(document_store, "_collection", None)
    if collection is None:
        return 0

    masks = parse_file_masks(file_mask)
    try:
        collection_data = collection.get(include=["metadatas"])
    except Exception as exc:
        st.session_state.document_status = f"Could not prune document index: {exc}"
        return 0

    ids_to_delete = []
    for document_id_value, metadata in zip(
        collection_data.get("ids", []),
        collection_data.get("metadatas", []),
    ):
        if not metadata or metadata.get("source") != "project_documents":
            continue
        source_path = metadata.get("source_path", "")
        if source_path and not matches_source_path(source_path, masks):
            ids_to_delete.append(document_id_value)

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return len(ids_to_delete)


def build_document_chunks(
    relative_path: str,
    content: str,
) -> tuple[list[str], list[dict[str, str | int]], list[str]]:
    file_hash = hashlib.sha256(
        f"{relative_path}\n{content}".encode("utf-8")
    ).hexdigest()
    chunks = chunk_text(content)
    texts = [build_document_metadata_text(relative_path)]
    metadatas = [
        {
            "source": "project_documents",
            "source_path": relative_path,
            "chunk_index": 0,
            "chunk_kind": "metadata",
            "file_hash": file_hash,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    ids = [document_id(relative_path, file_hash, 0)]

    for chunk_index, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append(
            {
                "source": "project_documents",
                "source_path": relative_path,
                "chunk_index": chunk_index + 1,
                "chunk_kind": "content",
                "file_hash": file_hash,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        ids.append(document_id(relative_path, file_hash, chunk_index + 1))

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


def format_index_progress(current: int, total: int, source_path: str) -> str:
    percentage = round((current / total) * 100) if total else 0
    return f"Indexing {source_path}: {current} of {total} ({percentage}%)"


def update_index_progress(current: int, total: int, source_path: str) -> str:
    progress_text = format_index_progress(current, total, source_path)
    st.session_state.index_progress_status = progress_text
    progress_value = current / total if total else 0.0
    st.session_state.index_progress_value = min(1.0, max(0.0, progress_value))
    return progress_text


def index_uploaded_document_payloads(
    uploaded_payloads: list[tuple[str, bytes]],
    progress_callback=None,
) -> tuple[int, int]:
    document_store = get_chroma_store(
        session_key="document_store",
        collection_name="project_documents",
        status_key="document_status",
        status_label="Document memory",
    )
    if document_store is None:
        return 0, 0

    indexed_files = 0
    indexed_chunks = 0
    skipped_files = []
    uploaded_sources = []
    total_files = len(uploaded_payloads)

    for file_index, (file_name, content) in enumerate(uploaded_payloads, start=1):
        update_index_progress(file_index, total_files, file_name)
        if progress_callback:
            progress_callback(file_index, total_files, file_name)

        suffix = Path(file_name).suffix.lower()
        if suffix not in DOCS_EXTENSIONS:
            skipped_files.append(file_name)
            continue

        try:
            text = read_uploaded_document_text(file_name, content)
        except (OSError, UnicodeDecodeError, zipfile.BadZipFile):
            skipped_files.append(file_name)
            continue

        chunk_count = add_document_to_store(document_store, file_name, text)
        if chunk_count:
            indexed_files += 1
            indexed_chunks += chunk_count
            uploaded_sources.append(file_name)

    if uploaded_sources:
        existing_sources = get_uploaded_document_sources()
        st.session_state.uploaded_document_sources = sorted(
            set(existing_sources + uploaded_sources)
        )
        st.session_state.uploaded_document_source = ", ".join(
            st.session_state.uploaded_document_sources
        )

    skipped_message = f"; skipped {len(skipped_files)} files" if skipped_files else ""
    st.session_state.document_status = (
        f"Indexed {indexed_files} uploaded files and {indexed_chunks} chunks"
        f"{skipped_message}"
    )
    return indexed_files, indexed_chunks


def index_uploaded_document(file_name: str, content: bytes) -> tuple[int, int]:
    return index_uploaded_document_payloads([(file_name, content)])


def index_documents(
    file_mask: str | None = None,
    progress_callback=None,
    prune_unmatched: bool = False,
) -> tuple[int, int]:
    document_store = get_document_store()
    if document_store is None:
        return 0, 0

    source_label = DOCS_FILE or DOCS_DIR or "document source"
    active_mask = file_mask if file_mask is not None else get_active_file_mask()
    pruned_chunks = (
        prune_document_chunks_outside_mask(document_store, active_mask)
        if prune_unmatched
        else 0
    )
    document_sources = list(iter_document_sources(active_mask))
    indexed_files = 0
    indexed_chunks = 0
    total_files = len(document_sources)

    if total_files == 0:
        st.session_state.index_progress_status = (
            f"No files matched mask {active_mask}"
        )
        st.session_state.index_progress_value = 0.0

    for file_index, (path, relative_path) in enumerate(document_sources, start=1):
        update_index_progress(file_index, total_files, relative_path)
        if progress_callback:
            progress_callback(file_index, total_files, relative_path)

        if path.suffix.lower() not in DOCS_EXTENSIONS:
            continue

        try:
            content = read_document_text(path)
        except (OSError, zipfile.BadZipFile):
            continue

        chunk_count = add_document_to_store(document_store, relative_path, content)
        if chunk_count:
            indexed_files += 1
            indexed_chunks += chunk_count

    pruned_message = (
        f"; removed {pruned_chunks} stale chunks outside mask"
        if pruned_chunks
        else ""
    )
    st.session_state.document_status = (
        f"Indexed {indexed_files} files and {indexed_chunks} chunks from "
        f"{source_label} using mask {active_mask}{pruned_message}"
    )
    return indexed_files, indexed_chunks


def document_source_signature(file_mask: str | None = None) -> str:
    uploaded_source = ",".join(get_uploaded_document_sources())
    extensions = ",".join(sorted(DOCS_EXTENSIONS))
    active_mask = file_mask if file_mask is not None else get_active_file_mask()
    return (
        f"{DOCS_DIR or ''}|{DOCS_FILE or ''}|{uploaded_source}|"
        f"{extensions}|{active_mask}"
    )


def ensure_documents_indexed_for_tool(file_mask: str | None = None) -> None:
    if not DOCS_DIR and not DOCS_FILE:
        return

    signature = document_source_signature(file_mask)
    if st.session_state.get("documents_indexed_for_tool") == signature:
        return

    try:
        index_documents(file_mask)
    except Exception as exc:
        st.session_state.document_status = f"Document tool auto-index failed: {exc}"
        return

    st.session_state.documents_indexed_for_tool = signature


@tool
def search_local_documents(query: str, file_mask: str = "") -> str:
    """
    Use this tool when the answer requires information from local project files,
    PDF reports, or DOCX documents indexed from DOCS_DIR, DOCS_FILE, or an
    uploaded file. Pass a concise semantic search query. Optionally pass
    file_mask to search only files matching a glob-style mask such as *.pdf,
    *.docx, reports/*.pdf, or reports/**/*.docx. The tool returns the most
    relevant text fragments with source labels.
    """
    active_mask = get_effective_tool_file_mask(file_mask)
    ensure_documents_indexed_for_tool(active_mask)
    return retrieve_document_context(query, active_mask)


@tool
def get_document_index_status(file_mask: str = "") -> str:
    """
    Use this tool when the user asks how many files are indexed, which files are
    indexed, what the current indexing progress is, or asks for document memory
    status. Optionally pass file_mask to count only matching files.
    """
    active_mask = get_effective_tool_file_mask(file_mask)
    snapshot = get_document_index_snapshot(active_mask)
    return format_document_index_snapshot(snapshot)


DOCUMENT_TOOLS = [search_local_documents, get_document_index_status]
DOCUMENT_TOOLS_BY_NAME = {
    document_tool.name: document_tool for document_tool in DOCUMENT_TOOLS
}


def extract_message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content)


def replay_text(text: str, estimator: TokenRateEstimator):
    for word in re.findall(r"\S+\s*", text):
        yield word
        time.sleep(estimator.delay_seconds)


def stream_chat_response(model, messages, estimator: TokenRateEstimator):
    previous_chunk_at = time.perf_counter()
    streamed_chunks = 0

    try:
        for chunk in model.stream(messages):
            chunk_text = extract_message_text(chunk)
            if not chunk_text:
                continue

            now = time.perf_counter()
            estimator.update(estimate_token_count(chunk_text), now - previous_chunk_at)
            previous_chunk_at = now
            streamed_chunks += 1

            yield chunk_text
    except NotImplementedError:
        streamed_chunks = 0

    if streamed_chunks > 0:
        return

    response_started_at = time.perf_counter()
    response_message = model.invoke(messages)
    response = extract_message_text(response_message)
    response_elapsed = time.perf_counter() - response_started_at
    estimator.update(estimate_token_count(response), response_elapsed)

    yield from replay_text(response, estimator)


def tool_call_value(tool_call, key: str, default=None):
    if isinstance(tool_call, dict):
        return tool_call.get(key, default)
    return getattr(tool_call, key, default)


def is_schema_placeholder(value) -> bool:
    return isinstance(value, dict) and set(value.keys()) <= {
        "type",
        "description",
        "title",
        "default",
    }


def normalize_tool_args(raw_args, fallback_query: str) -> dict[str, str]:
    if not isinstance(raw_args, dict):
        return {"query": str(raw_args or fallback_query), "file_mask": ""}

    query = raw_args.get("query")
    if not isinstance(query, str) or is_schema_placeholder(query) or not query.strip():
        query = fallback_query

    file_mask = raw_args.get("file_mask", "")
    if not isinstance(file_mask, str) or is_schema_placeholder(file_mask):
        file_mask = ""

    return {"query": query.strip(), "file_mask": file_mask.strip()}


def normalize_status_tool_args(raw_args) -> dict[str, str]:
    if not isinstance(raw_args, dict):
        return {"file_mask": ""}

    file_mask = raw_args.get("file_mask", "")
    if not isinstance(file_mask, str) or is_schema_placeholder(file_mask):
        file_mask = ""

    return {"file_mask": file_mask.strip()}


def normalize_args_for_tool(tool_name: str, raw_args, fallback_query: str) -> dict[str, str]:
    if tool_name == "get_document_index_status":
        return normalize_status_tool_args(raw_args)

    return normalize_tool_args(raw_args, fallback_query)


def describe_args_for_tool(tool_name: str, raw_args, fallback_query: str) -> dict[str, str]:
    args = normalize_args_for_tool(tool_name, raw_args, fallback_query)
    if "file_mask" not in args:
        return args

    requested_file_mask = args.get("file_mask", "")
    effective_file_mask = get_effective_tool_file_mask(requested_file_mask)
    display_args = dict(args)
    display_args["file_mask"] = effective_file_mask
    if requested_file_mask and requested_file_mask != effective_file_mask:
        display_args["requested_file_mask"] = requested_file_mask
    return display_args


def run_tool_call(tool_call, fallback_query: str) -> ToolMessage:
    tool_name = tool_call_value(tool_call, "name")
    tool_args = tool_call_value(tool_call, "args", {}) or {}
    tool_call_id = tool_call_value(tool_call, "id") or str(uuid4())
    selected_tool = DOCUMENT_TOOLS_BY_NAME.get(tool_name)

    if selected_tool is None:
        return ToolMessage(
            content=f"Tool {tool_name} is not available.",
            tool_call_id=tool_call_id,
        )

    tool_args = normalize_args_for_tool(tool_name, tool_args, fallback_query)

    try:
        tool_result = selected_tool.invoke(tool_args)
    except Exception as exc:
        tool_result = f"Tool {tool_name} failed: {exc}"

    return ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)


def build_agent_messages(user_input: str):
    return [
        SystemMessage(
            content=AGENT_SYSTEM_TEMPLATE.format(
                memory_context=retrieve_memory_context(user_input),
                recent_conversation=format_recent_conversation(),
            )
        ),
        HumanMessage(content=user_input),
    ]


def fallback_response_generator(user_input: str, estimator: TokenRateEstimator):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = OllamaLLM(model=OLLAMA_MODEL)
    chain = prompt | model | StrOutputParser()
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

    yield from replay_text(response, estimator)


def response_generator(user_input):
    estimator = get_token_rate_estimator()

    if is_recent_application_question(user_input):
        response = format_recent_applications_response(
            get_recent_application_records(get_active_file_mask())
        )
        estimator.update(estimate_token_count(response), 1)
        yield from replay_text(response, estimator)
        return

    if is_index_status_question(user_input):
        response = format_document_index_snapshot(
            get_document_index_snapshot(get_active_file_mask())
        )
        estimator.update(estimate_token_count(response), 1)
        yield from replay_text(response, estimator)
        return

    try:
        model = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        model_with_tools = model.bind_tools(DOCUMENT_TOOLS)
        messages = build_agent_messages(user_input)

        decision_started_at = time.perf_counter()
        decision_message = model_with_tools.invoke(messages)
        decision_elapsed = time.perf_counter() - decision_started_at
        tool_calls = getattr(decision_message, "tool_calls", []) or []

        if tool_calls:
            st.session_state.last_tool_calls = [
                {
                    "name": tool_call_value(tool_call, "name", "unknown"),
                    "args": describe_args_for_tool(
                        tool_call_value(tool_call, "name", "unknown"),
                        tool_call_value(tool_call, "args", {}),
                        user_input,
                    ),
                }
                for tool_call in tool_calls
            ]
            tool_messages = [
                run_tool_call(tool_call, user_input) for tool_call in tool_calls
            ]
            final_messages = messages + [decision_message] + tool_messages
            yield from stream_chat_response(model, final_messages, estimator)
            return

        response = extract_message_text(decision_message)
        estimator.update(estimate_token_count(response), decision_elapsed)
        yield from replay_text(response, estimator)
    except Exception as exc:
        st.session_state.document_status = (
            f"Agent tool calling unavailable: {exc}. Falling back to direct RAG."
        )
        yield from fallback_response_generator(user_input, estimator)


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
    uploaded_documents = st.file_uploader(
        "Choose documents",
        type=sorted(extension.lstrip(".") for extension in DOCS_EXTENSIONS),
        accept_multiple_files=True,
    )
    if "docs_file_mask" not in st.session_state:
        st.session_state.docs_file_mask = DOCS_FILE_MASK

    file_mask = st.text_area(
        "File masks",
        key="docs_file_mask",
        height=72,
        help="Use glob masks separated by commas or new lines, e.g. *.pdf, reports/**/*.docx",
    )
    progress_status = st.empty()
    progress_bar = st.progress(
        st.session_state.get("index_progress_value", 0.0),
    )
    progress_status.caption(
        st.session_state.get("index_progress_status", "Index progress: idle")
    )

    def sidebar_progress_callback(current: int, total: int, source_path: str) -> None:
        progress_text = format_index_progress(current, total, source_path)
        progress_value = current / total if total else 0.0
        progress_status.caption(progress_text)
        progress_bar.progress(min(1.0, max(0.0, progress_value)))

    if uploaded_documents:
        if st.button("Index selected files"):
            with st.spinner("Indexing selected files..."):
                index_uploaded_document_payloads(
                    [
                        (uploaded_document.name, uploaded_document.getvalue())
                        for uploaded_document in uploaded_documents
                    ],
                    sidebar_progress_callback,
                )
            st.rerun()

    st.caption(f"DOCS_DIR: {DOCS_DIR or 'not set'}")
    st.caption(f"DOCS_FILE: {DOCS_FILE or 'not set'}")
    st.caption(f"DOCS_FILE_MASK: {DOCS_FILE_MASK}")
    st.caption(f"RAG dir: {MEMORY_DIR}")
    st.caption(f"Extensions: {', '.join(sorted(DOCS_EXTENSIONS))}")
    st.caption(f"Results: {DOCS_RESULTS}")
    st.caption("Agent tools: search_local_documents, get_document_index_status")
    last_tool_calls = st.session_state.get("last_tool_calls")
    if last_tool_calls:
        st.caption(f"Last tool call: {last_tool_calls[-1]}")
    if st.button(
        "Reindex configured source by mask",
        disabled=not bool(DOCS_DIR or DOCS_FILE),
    ):
        with st.spinner("Indexing documents..."):
            index_documents(file_mask, sidebar_progress_callback, prune_unmatched=True)
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
