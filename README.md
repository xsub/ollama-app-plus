# ollama-app-plus

# Download and install ollama
https://ollama.com

# Run a model locally

Choose a model from the library: https://ollama.com/library

Run it locally:

```ollama run <MODEL_NAME>```

Download the local embedding model used for long-term memory:

```ollama pull nomic-embed-text```

# Install pip packages

Install pip packages with:
```pip install -r requirements.txt```

# Choose a model

The app uses `SpeakLeash/bielik-7b-instruct-v0.1-gguf` by default. To use another
local Ollama model, set `MODEL_NAME` when starting Streamlit:

```MODEL_NAME=<MODEL_NAME> streamlit run app.py```

For example, after downloading `llama3.2` with `ollama run llama3.2`, start the
app with:

```MODEL_NAME=llama3.2 streamlit run app.py```

# Run your application

Run the application with the default model:

```streamlit run app.py```

# Adaptive response pacing

The app adapts its response pacing to the currently selected Ollama model and
machine performance. This replaces the previous fixed `sleep(0.05)` delay with a
small feedback loop that estimates response throughput while the model is being
used.

How it works:

1. The response generator first tries to use LangChain/Ollama streaming with
   `chain.stream(...)`.
2. For each streamed chunk, the app measures elapsed time with `time.perf_counter()`
   and estimates how many text tokens were received.
3. The observed tokens-per-second value is smoothed with an exponentially weighted
   moving average, so a single slow or fast response does not over-adjust pacing.
4. The current estimate is stored in Streamlit session state and reused by later
   requests in the same browser session.
5. If streaming is unavailable and the app must fall back to a full `chain.invoke(...)`
   response, the replay delay is calculated from the measured token rate instead
   of using a hard-coded constant.

The fallback delay is derived as:

```python
delay_seconds = 1 / estimated_tokens_per_second
```

The value is clamped between `MIN_TOKEN_DELAY_SECONDS` and
`MAX_TOKEN_DELAY_SECONDS`, so very fast models do not produce an excessively tight
UI loop and very slow models do not make the replay feel stalled. The UI displays
the current estimated token rate and fallback delay above the chat input.

In normal operation with streaming enabled, chunks are displayed at the model's
native pace. The adaptive delay mostly matters for fallback replay, while the same
measurements still provide a useful live benchmark of the selected model and
hardware setup.

# Long-term conversation memory

The app can keep a local semantic memory of previous exchanges by using Chroma as
a file-based vector database and Ollama embeddings. This lets the model retrieve
older relevant conversation fragments without sending the whole dialog back to the
LLM on every request.

How it works:

1. The current browser session keeps recent messages in Streamlit session state.
2. Before answering a new prompt, the app embeds the user question with
   `nomic-embed-text`.
3. Chroma searches `./memory_db` for the most similar previous exchanges.
4. The prompt receives three context blocks: relevant long-term memory, recent
   conversation, and the current question.
5. After the assistant response is complete, the user/assistant exchange is saved
   back to Chroma for future retrieval.

The memory database is stored locally in `./memory_db` and is ignored by Git.

Useful environment variables:

```bash
EMBEDDING_MODEL=nomic-embed-text
MEMORY_DIR=./memory_db
MEMORY_RESULTS=4
RECENT_MESSAGES_LIMIT=6
```

For example:

```bash
MODEL_NAME=llama3.2 EMBEDDING_MODEL=nomic-embed-text streamlit run app.py
```
