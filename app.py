import streamlit as st
import time
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None

DEFAULT_OLLAMA_MODEL = "SpeakLeash/bielik-7b-instruct-v0.1-gguf"
OLLAMA_MODEL = os.getenv("MODEL_NAME", DEFAULT_OLLAMA_MODEL)
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
MEMORY_DIR = os.getenv("MEMORY_DIR", "./memory_db")
MEMORY_RESULTS = int(os.getenv("MEMORY_RESULTS", "4"))
RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "6"))
PROMPT_TEMPLATE = """
Jestes pomocnym asystentem.

Relevant long-term memory:
{memory_context}

Recent conversation:
{recent_conversation}

Current user question:
{question}

Odpowiedz na pytanie, korzystajac z pamieci tylko wtedy, gdy jest istotna.
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
    if Chroma is None:
        st.session_state.memory_status = (
            "Long-term memory disabled: install langchain-chroma."
        )
        return None

    if "memory_store" not in st.session_state:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        st.session_state.memory_store = Chroma(
            collection_name="conversation_memory",
            persist_directory=MEMORY_DIR,
            embedding_function=embeddings,
        )

    st.session_state.memory_status = (
        f"Long-term memory: {MEMORY_DIR} via {EMBEDDING_MODEL}"
    )
    return st.session_state.memory_store


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


def response_generator(user_input):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    model = OllamaLLM(model=OLLAMA_MODEL)

    chain = prompt | model | StrOutputParser()
    estimator = get_token_rate_estimator()
    chain_input = {
        "memory_context": retrieve_memory_context(user_input),
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

estimator = get_token_rate_estimator()
st.caption(
    f"Estimated token rate: {estimator.tokens_per_second:.1f} tokens/s "
    f"(fallback delay: {estimator.delay_seconds:.3f}s)"
)
st.caption(st.session_state.get("memory_status", "Long-term memory initializing."))

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
