import streamlit as st
import time
import os
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

DEFAULT_OLLAMA_MODEL = "SpeakLeash/bielik-7b-instruct-v0.1-gguf"
OLLAMA_MODEL = os.getenv("MODEL_NAME", DEFAULT_OLLAMA_MODEL)
PROMPT_TEMPLATE = """Odpowiedz na pytanie: {question}"""
CHAT_TITLE = "Ollama chat"
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


def response_generator(user_input):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    model = OllamaLLM(model=OLLAMA_MODEL)

    chain = prompt | model | StrOutputParser()
    estimator = get_token_rate_estimator()

    previous_chunk_at = time.perf_counter()
    streamed_chunks = 0

    try:
        for chunk in chain.stream({"question": user_input}):
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
    response = chain.invoke({"question": user_input})
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

# Accept user input
if user_input := st.chat_input(CHAT_HINT):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(user_input))
