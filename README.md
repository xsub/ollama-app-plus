# streamlit-langchain-ai-agent-rag-experiment

Local-first Streamlit + LangChain AI Agent RAG experiment with Ollama, Chroma,
tool calling, local PDF/DOCX indexing, semantic document memory, and an
observable retrieval loop.

`streamlit-langchain-ai-agent-rag-experiment` is a practical playground for
building an agentic RAG assistant that runs locally. It combines a Streamlit chat
interface, LangChain tool binding, Ollama models and embeddings, Chroma vector
storage, and document-aware retrieval so the assistant can decide when to search
local files instead of answering from model memory.

The project is designed as a compact but ambitious reference implementation of
modern AI agent architecture: local LLM inference, vector search, tool calling,
ReAct-style decision flow, document ingestion, filename-aware metadata retrieval,
stateful conversation memory, and a UI that makes indexing and retrieval visible
while they run.

## Technology stack

- Streamlit for the interactive chat UI, sidebar controls, upload flow, and live
  indexing progress.
- Ollama for local model serving and private on-machine inference.
- LangChain and LangChain Ollama for prompt composition, streaming responses,
  chat model integration, tool binding, and agent-style control flow.
- ChromaDB via `langchain-chroma` for persistent local vector storage.
- Ollama embeddings with `nomic-embed-text` for semantic document and memory
  retrieval.
- `pypdf` for PDF text extraction.
- Python standard-library OOXML parsing for DOCX text extraction without an
  extra runtime dependency.
- Environment-driven configuration for model selection, document directories,
  file masks, chunk sizing, retrieval limits, and persistence paths.

## Core capabilities

- Local chat with selectable Ollama models.
- Long-term conversation memory stored in a local Chroma collection.
- Multi-file document upload and directory indexing.
- Glob-style file masks such as `*.pdf`, `*.docx`, or `reports/**/*.pdf`.
- Agent tools for local document search and document-index status.
- Metadata-aware retrieval that treats filenames as evidence for questions about
  CVs, applications, roles, companies, and dates.
- Live indexing progress, including current file, total file count, and
  percentage.
- Graceful fallback to direct RAG when tool calling is unavailable.

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
3. Chroma searches the local RAG directory for the most similar previous exchanges.
4. The prompt receives three context blocks: relevant long-term memory, recent
   conversation, and the current question.
5. After the assistant response is complete, the user/assistant exchange is saved
   back to Chroma for future retrieval.

The memory database is stored locally in `.OAP_RAG` by default and is ignored by
Git. When `DOCS_DIR` is set, the default RAG directory is created inside that
folder, for example `DOCS_DIR=./knowledge` stores Chroma data in
`./knowledge/.OAP_RAG`. You can override this with `MEMORY_DIR`.

Useful environment variables:

```bash
EMBEDDING_MODEL=nomic-embed-text
MEMORY_DIR=./.OAP_RAG
MEMORY_RESULTS=4
RECENT_MESSAGES_LIMIT=6
```

For example:

```bash
MODEL_NAME=llama3.2 EMBEDDING_MODEL=nomic-embed-text streamlit run app.py
```

# File-based document memory

The app can also index files from a local directory and use the most relevant
chunks as context for answers. This is a small local RAG flow for prompts such as
"read the files in this directory and include them in the answer".

To index documents from your machine, use the sidebar file picker and click
`Index selected files`. The picker accepts multiple files at once. This supports
text-like files, PDFs, and DOCX documents. Uploaded files are embedded and stored
in the local Chroma document collection; the files themselves are not copied into
the repo. While indexing, the sidebar shows the current file plus progress such
as `4 of 15 (27%)`.

Set `DOCS_DIR` when a whole directory should be indexed:

```bash
DOCS_DIR=./knowledge MODEL_NAME=llama3.2 streamlit run app.py
```

You can narrow directory indexing with glob-style masks in the sidebar `File
masks` field. Masks can be separated by commas or new lines, for example:

```bash
*.pdf
*.docx
reports/**/*.pdf
```

The sidebar mask is treated as the top-level safety filter. Agent tool calls
cannot broaden it; if the sidebar is set to `*CV*2026*.pdf`, a model-generated
tool mask such as `*.pdf` will still be constrained to the sidebar mask.

Or set `DOCS_FILE` when you want to index one file from the environment instead
of using the sidebar file picker:

```bash
DOCS_FILE=./manual.pdf MODEL_NAME=llama3.2 streamlit run app.py
```

The app adds a sidebar panel with a multi-file picker, an `Index selected files`
button, and a `Reindex configured source by mask` button. Reindexing reads
supported files, splits them into overlapping chunks, embeds the chunks with
`nomic-embed-text`, and stores them in Chroma under a separate
`project_documents` collection. The manual reindex action also prunes stale
document chunks that do not match the active sidebar mask, so accidental broad
tool calls do not leave the working document memory polluted. Conversation
memory and document memory share the same local Chroma persistence directory,
but use different collections. By default, that persistence directory is
`DOCS_DIR/.OAP_RAG` when `DOCS_DIR` is set, or a sibling `.OAP_RAG` directory
next to `DOCS_FILE` when a single file is used.

Supported file extensions default to:

```bash
.txt,.md,.py,.json,.yaml,.yml,.toml,.ini,.cfg,.pdf,.docx
```

Useful document memory environment variables:

```bash
DOCS_DIR=./knowledge
DOCS_FILE=./manual.pdf
MEMORY_DIR=./knowledge/.OAP_RAG
DOCS_RESULTS=6
DOCS_CHUNK_SIZE=1500
DOCS_CHUNK_OVERLAP=200
DOCS_FILE_MASK=*.pdf,*.docx
DOCS_EXTENSIONS=.txt,.md,.py,.json,.yaml,.yml,.toml,.ini,.cfg,.pdf,.docx
SUPPRESS_PDF_WARNINGS=1
```

For PDFs, the app extracts text from every page, labels chunks with page markers
where possible, then indexes the extracted text through the same Chroma document
memory flow. For DOCX files, the app extracts text directly from the Office Open
XML document body, headers, and footers. Every indexed file also gets a metadata
chunk containing the source path and a readable version of the filename, so names
like `Pawel_Suchanecki_CV_VyOS_Development_Manager_2026-07.pdf` can be used as
evidence when the user asks about the indexed CV, target company, role, or date.
Reindex existing documents after upgrading to create these filename metadata
chunks.

Some PDFs contain broken or unusual cross-reference tables. `pypdf` can still
extract text from many of them, but may print noisy warnings such as
`Ignoring wrong pointing object ...`. These warnings are suppressed by default
during PDF parsing. Set `SUPPRESS_PDF_WARNINGS=0` when you want to debug PDF
parser internals.

## Agentic document search tool

The app now exposes the document retriever as an agent tool named
`search_local_documents` and a status tool named `get_document_index_status`.
Instead of always injecting document context into every prompt, the chat flow
first asks a tool-capable Ollama chat model whether it needs local files to
answer the current question.

The decision loop is:

1. The user asks a question.
2. `ChatOllama.bind_tools(...)` gives the model access to
   `search_local_documents` and `get_document_index_status`.
3. If the model returns a tool call, the Streamlit app runs the local Chroma
   search against indexed files from `DOCS_DIR`, `DOCS_FILE`, or uploaded
   documents. If the user asks for a file mask such as `*.pdf`, the model can
   pass it as the tool's optional `file_mask` argument.
4. The search result is sent back as a `ToolMessage`.
5. The model writes the final natural-language answer using the returned
   fragments and source labels.

When `DOCS_DIR` or `DOCS_FILE` is configured, the tool auto-indexes that source
the first time it is used in the Streamlit session. The sidebar still keeps the
manual `Reindex configured source` button for explicit refreshes.

Use a model that supports Ollama tool calling for the agent path, for example:

```bash
MODEL_NAME=qwen3 DOCS_DIR=./knowledge streamlit run app.py
```

If the selected model or local Ollama setup does not support tool calling, the
app falls back to the previous direct RAG prompt flow.

Questions about index state, such as "how many files are indexed?", are answered
directly from Chroma metadata so the app does not ask the LLM to infer counts
from retrieved document text.

Questions about recent applications, such as "where did Pawel apply most
recently and for what role?", are also handled from indexed filenames first. The
app parses dates, company hints, and role hints from names like
`Pawel_Suchanecki_CV_VyOS_Development_Manager_2026-07.pdf` before falling back
to semantic document search.

In the fallback flow, the prompt receives:

1. relevant long-term conversation memory,
2. relevant project file chunks,
3. recent conversation,
4. the current user question.

The indexed file chunks include source path metadata, so retrieved context is
passed to the model with source labels such as `Source: README.md chunk 0`.
