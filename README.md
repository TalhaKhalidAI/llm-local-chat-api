# FastAPI Local LLM Chat API, Multimodal AI Orchestrator & Authentication System

A production-grade FastAPI local LLM serving platform featuring **True Iteration-Level Continuous Batching**, a **Multimodal AI Orchestrator** (Persistent ChromaDB RAG, Live Web Search + YouTube Lookup, and 100% In-RAM Vision/OCR via `Qwen2-VL-2B`), centralized exception handling, SQLite database, JWT authentication, and role-based access control.

## Made by Talha Khalid

---

## Features

### 1. True Iteration-Level Continuous Batching LLM Engine (`App/core/AsyncLLM.py`)
- **Single Shared Model + Multi-Sequence KV Cache (`n_seq_max`)**: Loads GGUF model weights **once** into VRAM/RAM and partitions a unified `LlamaContext` across `LLM_CONTINUOUS_BATCH_SLOTS` concurrent sequences (`seq_id = 0 .. N-1`) via `llama_batch_init` and `llama_decode`.
- **Zero-Wait Dynamic Slot Admission**: New requests immediately enter freed sequence slots between token steps without waiting for running requests to finish.
- **Native GGUF Chat Templates (`format_messages_native`)**: Renders prompts using the model's embedded Jinja2 `tokenizer.chat_template` (supporting Llama-3, GLM-4, DeepSeek-V2, Qwen2.5, Gemma, Phi, etc.).
- **Grammar-Enforced Structured Output & Tool Calling**: Supports OpenAI-compatible `response_format` (`json_object` and `json_schema` compiled to C++ GBNF grammars), `tools`, `tool_choice`, and structured `tool_calls`.
- **Exact Token Accounting & Generation Budget Reservation**: Tracks exact `prompt_tokens`, `completion_tokens`, and `total_tokens` (including `stream_options.include_usage` on SSE streams) and reserves generation token headroom so long RAG/web prompts never truncate answers prematurely.
- **Client Disconnect Cancellation**: Automatically aborts token generation and frees the KV sequence slot (`llama_memory_seq_rm`) when a streaming client disconnects.
- **Hardcoded Model Stop Tokens Matrix & Special Control Token Handling**: Defines per-model native stop tokens across all 11 supported local models (`GLM-4`, `DeepSeek-V2-Lite`, `DeepSeek-Coder-7B`, `Meta-Llama-3.1-8B`, `Qwen2.5-Coder-7B`, `Qwen2-VL-2B`, `Gemma-2-9B`, `SmolVLM-500M`, `OpenHermes-2.5`). Uses `detokenize([tok], special=True)` during continuous batching so special control tokens (such as `<|user|>`, `<|observation|>`, `<|im_end|>`) are cleanly detected without hanging.
- **Explicit Model Lifecycle (No Unwanted Auto-Loading)**: Chat and Vision endpoints strictly require a model to be loaded via `POST /app/v1/load_model` first, returning a structured `MODEL_NOT_LOADED` (`400`) error if no model is active.

### 2. Multimodal AI Orchestrator (`App/core/orchestrator.py`)
- **Strict Multi-Tenant Isolation (`user_id` & `session_id`)**:
  - Full cryptographic and vector isolation for all ChromaDB collections (`chat_history`, `docs_rag`, `web_cache`, `image_index`) and in-RAM buffers.
  - Automatic tenant resolution cascade: Authenticated JWT Claims -> Request Body/Form fields -> HTTP Headers (`X-User-ID`, `X-Session-ID`) -> Auto-generated deterministic random tenant UUIDs (`user_<hex>`, `sess_<hex>`).
  - Tenant IDs are returned in all completion responses (`user_id`, `session_id`) to ensure frontends can persist conversations seamlessly.
- **Adaptive 4-Profile Document Indexing Engine (`UniversalDocumentParser`)**:
  - **100% In-RAM Processing**: Strictly operates in system RAM—never writes uploaded PDFs, Office docs, spreadsheets, or archives to disk.
  - **Automatic Profile Classification**:
    1. `datasheet_or_table`: Detects tabular/datasheet formats and automatically replicates markdown table headers across every chunk with semantic key-value serialization.
    2. `mixed_doc_with_tables`: Hybrid chunking isolating narrative prose from embedded data tables.
    3. `structured_code_or_json`: Syntax-preserving block chunking for config, schema, and source files.
    4. `plain_narrative`: Overlapping semantic paragraph chunking for prose, books, and articles.
  - **Broad Format Support**: PDF (with automatic scanned-page OCR via `Qwen2-VL-2B`), DOCX, DOC, XLSX, CSV, TSV, ODT, EPUB, RTF, JSON, XML, HTML, ZIP/TAR, and source code.
- **Smart Best-1 Memory Winner Selection with Refusal & Anti-Poisoning Guard**:
  - Scores candidate memories using semantic cosine distance + turn length + factual boost.
  - Filter out refusals, apologies, and hallucinations (`"I cannot"`, `"as an AI"`, `"I apologize"`) so unhelpful or toxic past turns never poison the active prompt context.
- **Live Internet Search & Direct YouTube Subtitle Lookup**:
  - Multi-tier search using DuckDuckGo (`ddgs` / HTML) with automatic fallback to Bing News RSS & Google News RSS.
  - Direct YouTube Lookup: Parses `youtube.com/watch` URLs, extracts video metadata, and ingests timedtext video captions/transcripts for full video RAG.
  - Query sanitizer that strips conversational filler and corrects common typos before querying search engines.
  - SSRF-safe asynchronous HTML scraper (`_is_safe_public_url` blocking loopback/private/link-local IPs).
- **100% In-RAM Photo Describer & OCR (`Qwen2-VL-2B-Instruct` + `mmproj`)**:
  - Processes uploaded images (`PNG`, `JPEG`, `WEBP`, `GIF`) **strictly in RAM**—never writes uploaded files to disk.
  - Pipes image bytes through ImageMagick (`magick`) in RAM to downscale large photos to `<=768x768` (~567 vision tokens), preventing CPU/RAM spikes.
  - Bounded `asyncio.Queue(maxsize=10)` with sticky batching and CPU-only execution (`ORCH_VISION_GPU_LAYERS=0`) so the active chat LLM stays 100% resident in GPU VRAM.
  - Enforces a strict GBNF JSON schema (`VisionReport`: `scene`, `objects`, `subjects_and_details`, `colors_and_lighting`, `visible_text`, `summary`, `tags`, `search_query`).

### 3. Polyglot AST Coding & Repository Engine (`POST /app/v1/chat/completions/code`)
- **Polyglot Syntax & Symbol Extraction (`UniversalCodeASTProcessor`)**: Native AST parsing and regex symbol extraction for Python, JavaScript/TypeScript, Rust, Go, Java, C/C++, PHP, Ruby, and Shell scripts.
- **Token-Budget AST Skeleton Compression**: Generates high-level file skeletons (classes, functions, method signatures, exports) and selectively injects relevant AST nodes matching the user's coding query, enabling large files and multi-file codebases to fit comfortably within compact LLM context windows without losing critical structural context.

### 4. Centralized Exception Architecture & Recovery (`main.py` & `App/core/exceptions.py`)
- All application, validation, and runtime errors are caught centrally in `main.py` via FastAPI `@app.exception_handler(...)` decorators.
- Custom exception hierarchy (`AppError`, `ModelNotLoadedError`, `ModelNotFoundError`, `ModelLoadError`, `QueueFullError`, `InvalidImageError`, `ImageTooLargeError`, `InvalidCollectionError`, `MissingRequestFieldError`, `InsufficientMemoryError`) returning uniform `{"error": "<CODE>", "detail": "..."}` JSON responses.
- **Resilient Kill-Switch & Recovery**:
  - Protected static assets (`/assets`) and health endpoints from inadvertent lockouts.
  - Safe `jsonable_encoder` serialization preventing non-serializable validation exceptions from tripping the maintenance kill switch.
  - Instant recovery endpoint: `POST /reset-kill-switch` to clear the emergency kill switch without server restarts.

### 4. Authentication, User Management & Security
- JWT authentication with Argon2 password hashing.
- Role-based access control (`admin` and `user` roles).
- Profile management, password change/reset, account activation/deactivation, and soft delete with restore capability.
- Rate limiting via SlowAPI and global maintenance kill-switch middleware.

---

## Tech Stack

- **Framework**: FastAPI
- **LLM Runtime**: `llama-cpp-python` (GGUF / Vulkan / CUDA / CPU)
- **Vector Database**: ChromaDB (Persistent local disk)
- **Vision / Multimodal**: `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` + `mmproj-Qwen2-VL-2B-Instruct-f16.gguf`
- **Database**: SQLite (via SQLAlchemy)
- **Authentication**: JWT with Argon2 password hashing
- **Validation**: Pydantic v2
- **Rate Limiting**: SlowAPI

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/iot-Noob/API-Boiler-Plate.git
cd API-Boiler-Plate
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Configure `.env`
```env
# Security (REQUIRED)
SECRET_KEY=your-secret-key-here-min-32-chars-long!
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=790

# Database & Logging
SQLITE_DATABASE_URL=sqlite:///./app.db
LOG_FILEPATH=./logs/
DEFAULT_TIMEOUT=18000
RATE_LIMIT_DEFAULT=100/minute
KILL_SWITCH_ENABLED=false

# Argon2 Hashing
MEMORY_COST=65536
PARALLELISM=2
HASH_LENGTH=32
SALT_LENGTH=16

# Default Admin User
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=Admin@123

# Environment
ENVIRONMENT=development
ALLOWED_ORIGINS=*

# ==========================================
# LLM ENGINE & CONTINUOUS BATCHING SETTINGS
# ==========================================
LLM_MODEL_PATH=/mnt/llm_models/LLM_Models
LLM_N_CTX=4096
LLM_N_GPU_LAYERS=44
LLM_N_BATCH=128
LLM_N_THREADS=6
LLM_VERBOSE=true
USE_MMAP=true
USE_MLOCK=false
LLM_CONTINUOUS_BATCH_SLOTS=2
LLM_MIN_AVAILABLE_RAM_MB=3000
STREAM_RESPONSE=true

# Default Sampling Fallbacks
DEFAULT_TEMPERATURE=0.4
DEFAULT_TOP_P=0.9
DEFAULT_TOP_K=40
DEFAULT_MAX_TOKENS=1024
DEFAULT_REPEAT_PENALTY=1.1

# ==========================================
# ORCHESTRATOR / RAG / WEB SEARCH / VISION
# ==========================================
ORCH_ENABLED=true
ORCH_CHROMA_DIR=./chroma_db
ORCH_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ORCH_DDG_MAX_RESULTS=5
ORCH_WEB_CACHE_TTL_HOURS=24
ORCH_SCRAPE_TIMEOUT_SEC=5.0
ORCH_SCRAPE_MAX_BYTES=1500000
ORCH_CHUNK_SIZE_CHARS=900
ORCH_CHUNK_OVERLAP_CHARS=120
ORCH_MAX_RAG_CHUNKS=3
ORCH_MAX_DISTANCE=0.65
ORCH_VISION_MODEL=/mnt/llm_models/LLM_Models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf
ORCH_VISION_MMPROJ=/mnt/llm_models/LLM_Models/mmproj-Qwen2-VL-2B-Instruct-f16.gguf
ORCH_VISION_GPU_LAYERS=0
```

### 4. Run the application
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI is available offline at `http://localhost:8000/docs`.

---

## API Endpoints

### LLM Chat, Vision, Orchestrator & Model Management (`/app/v1`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/app/v1/load_model` | Hot-load a `.gguf` model into VRAM/RAM (required before calling chat/vision endpoints) |
| POST | `/app/v1/unload_model` | Unload the active model and release VRAM/RAM |
| GET | `/app/v1/active_model` | Inspect the currently loaded model, continuous batching slots, and memory usage |
| GET | `/app/v1/models` | List all `.gguf` models available in `LLM_MODEL_PATH` |
| POST | `/app/v1/vision/load_model` | Load a Vision GGUF model + mmproj projector into memory (kept resident for ~1.5s rapid OCR/inference) |
| POST | `/app/v1/vision/unload_model` | Unload resident Vision model and free system memory |
| GET | `/app/v1/vision/models` | List available vision `.gguf` models and `mmproj` projectors strictly in `VISION_MODEL_PATH` |
| GET | `/app/v1/vision/active_model` | Get status, active model/mmproj names, and paths for currently resident vision model |
| POST | `/app/v1/chat/completions` | OpenAI-compatible chat completion supporting JSON or `multipart/form-data` (`request` JSON + optional `image`/document upload), strict tenant isolation (`user_id`, `session_id`), RAG, live web search, tool calls, GBNF structured outputs, and SSE streaming |
| POST | `/app/v1/vision/describe` | Upload an image (processed 100% in RAM) to get a structured `VisionReport` + OCR and index it in ChromaDB |
| POST | `/app/v1/chat/completions/upload` | Dedicated multipart upload endpoint for chatting with uploaded files/images/documents (PDF, DOCX, XLSX, CSV, ZIP, Code) with chained JSON `system_prompts` and tenant isolation |
| POST | `/app/v1/chat/completions/code` | Dedicated Polyglot AST Coding endpoint with AST skeleton extraction, symbol indexing, and token-saving compression for large source files |
| GET | `/app/v1/web_search/test` | Diagnostic endpoint to test live web search & ChromaDB `web_cache` ingestion |
| GET | `/app/v1/history` | List stored entries from any ChromaDB collection (`chat_history`, `web_cache`, `image_index`, `docs_rag`) |
| GET | `/app/v1/history/search` | Semantic vector search across any ChromaDB collection |
| GET | `/app/v1/health` | LLM engine health check, queue depth, and memory metrics |
| POST | `/reset-kill-switch` | Instant recovery endpoint to clear emergency kill switch without server restart |

### Authentication & User Management (`/app/v1/users`)
| Method | Endpoint | Description | Access |
|--------|----------|-------------|--------|
| POST | `/app/v1/users/signup` | Register a new user | Public |
| POST | `/app/v1/users/login` | Authenticate and obtain JWT token | Public |
| GET | `/app/v1/users/me` | Get current user profile | User |
| GET | `/app/v1/users/` | List all users | Admin |
| GET | `/app/v1/users/{id}` | Get user by ID | User (own), Admin (any) |
| PUT | `/app/v1/users/{id}` | Update user | User (own), Admin (any) |
| DELETE | `/app/v1/users/{id}` | Soft-delete user | User (own), Admin (any) |
| POST | `/app/v1/users/change-password` | Change own password | User |
| POST | `/app/v1/users/{id}/reset-password` | Reset user password | Admin |
| POST | `/app/v1/users/{id}/activate` | Activate user | Admin |
| POST | `/app/v1/users/{id}/deactivate` | Deactivate user | Admin |
| GET | `/app/v1/users/deleted/list` | List soft-deleted users | Admin |
| POST | `/app/v1/users/{id}/restore` | Restore soft-deleted user | Admin |

---

## Usage Examples

### 1. Load a Model First (Required)
```bash
curl -X POST "http://localhost:8000/app/v1/load_model?model_name=deepseek-v2-lite-chat-q4_k_m.gguf"
```

### 2. Chat Completion with Live Web Search (Multipart Form or JSON)
```bash
curl -X POST "http://localhost:8000/app/v1/chat/completions" \
  -H "Content-Type: multipart/form-data" \
  -F 'request={
    "model": "deepseek-v2-lite-chat-q4_k_m.gguf",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "What is the latest news about NVIDIA CEO?"}
    ],
    "temperature": 0.3,
    "max_tokens": 1024,
    "stream": false,
    "use_orchestrator": true,
    "use_web": true,
    "use_rag": false
  }'
```

### 3. Multimodal Chat with Image Upload (100% In-RAM Vision + Web Search)
```bash
curl -X POST "http://localhost:8000/app/v1/chat/completions" \
  -H "Content-Type: multipart/form-data" \
  -F 'request={
    "model": "deepseek-v2-lite-chat-q4_k_m.gguf",
    "messages": [
      {"role": "user", "content": "Describe the black hole in this image and search the internet about it."}
    ],
    "temperature": 0.2,
    "max_tokens": 1024,
    "use_orchestrator": true,
    "use_web": true
  }' \
  -F "image=@/path/to/photo.jpg;type=image/jpeg"
```

### 4. Standalone In-RAM Image Description & OCR
```bash
curl -X POST "http://localhost:8000/app/v1/vision/describe" \
  -F "image=@/path/to/image.png" \
  -F "prompt=Extract all visible text and describe the scene"
```

### 5. Vision Model Lifecycle (Resident In-RAM Loading)
```bash
# List available vision models in VISION_MODEL_PATH
curl -X GET "http://localhost:8000/app/v1/vision/models"

# Load Vision Model + mmproj into RAM (stays resident for ~1.5s rapid requests)
curl -X POST "http://localhost:8000/app/v1/vision/load_model" \
  -d "model_name=Qwen2-VL-2B-Instruct-Q4_K_M.gguf&mmproj_name=mmproj-Qwen2-VL-2B-Instruct-f16.gguf&gpu_layers=0"

# Inspect Active Vision Model
curl -X GET "http://localhost:8000/app/v1/vision/active_model"

# Unload Vision Model to free RAM
curl -X POST "http://localhost:8000/app/v1/vision/unload_model"
```

### 6. Cluster / Multi-Node Disaggregated Vision Architecture
For low-end hardware or GPU clusters, you can run Text LLMs on Node 1 and Vision on Node 2:
- On Node 1 (Text API), set in `.env`:
  ```bash
  VISION_API_URL=http://node2-vision-server:8000/app/v1/vision/describe
  ```
- When a user uploads an image, Node 1 checks if a local vision model is loaded. If not, it automatically forwards the image in RAM to Node 2, indexes the remote OCR result in local ChromaDB, and passes context directly to the active LLM.

### 7. Polyglot AST Coding & Repository Analysis
```bash
curl -X POST "http://localhost:8000/app/v1/chat/completions/code" \
  -F "code_file=@/path/to/repository_code.py;type=text/x-python" \
  -F "prompt=Optimize the memory allocation inside the AST processor" \
  -F "model=deepseek-coder-7b-instruct-v1.5-Q3_K_L.gguf" \
  -F "user_id=tenant_user_01" \
  -F "session_id=coding_sess_42"
```

### 8. Document & File Chat with Chained System Prompts
```bash
curl -X POST "http://localhost:8000/app/v1/chat/completions/upload" \
  -F "image=@/path/to/financial_datasheet.xlsx" \
  -F "prompt=What were the Q3 operating margins and net profit?" \
  -F 'system_prompts=["You are an expert financial analyst.", "Format all numbers as standard tables with USD currency symbols."]' \
  -F "model=glm-4-9b-chat-IQ4_XS.gguf" \
  -F "use_rag=true" \
  -F "user_id=fin_analyst" \
  -F "session_id=q3_report"
```

### 9. Emergency Reset for Kill Switch
```bash
curl -X POST "http://localhost:8000/reset-kill-switch"
```

---

## Project Structure

```text
llm-local-chat-api/
├── App/
│   ├── api/
│   │   ├── dependencies/
│   │   │   ├── auth.py              # JWT & Argon2 password utilities
│   │   │   └── sqlite_connector.py  # SQLAlchemy session management
│   │   └── v1/
│   │       ├── LlmChatEp.py         # Chat, Vision, History, Web Search & Model routes
│   │       └── Users.py             # User & Admin management routes
│   ├── core/
│   │   ├── AsyncLLM.py              # True Continuous Batching GGUF engine & LCEL Runnables
│   │   ├── orchestrator.py          # ChromaDB RAG, DuckDuckGo/RSS/YouTube Web Search & Qwen2-VL Vision
│   │   ├── exceptions.py            # Custom AppError exception hierarchy
│   │   ├── settings.py              # Pydantic Settings configuration
│   │   └── LoggingInit.py           # Rotating file & console logging
│   ├── models/
│   │   ├── llmModels.py             # OpenAI-compatible ChatRequest, ChatMessage, ChatResponse
│   │   └── userModels.py            # User Pydantic models
│   ├── repository/
│   │   ├── llmRepository.py         # AsyncLLM repository lifecycle & request dispatch
│   │   └── userRepository.py        # User SQLite CRUD operations
│   └── schemas/
│       └── userSchemas.py           # SQLAlchemy ORM models
├── tests/
│   └── test_orchestrator.py         # Pytest suite for RAG, Web Search, SSRF, Grounding & Vision
├── main.py                          # FastAPI entry point & centralized @app.exception_handler definitions
├── .env                             # Environment variables
└── app.db                           # SQLite database (auto-created)
```

---

## Environment Variables Reference

### LLM Engine & Continuous Batching
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_MODEL_PATH` | Yes | - | Directory containing `.gguf` models |
| `LLM_N_CTX` | No | `4096` | Total KV cache context window in tokens (divided across `LLM_CONTINUOUS_BATCH_SLOTS`) |
| `LLM_N_GPU_LAYERS` | No | `0` | Layers offloaded to GPU (`-1` or high value for full offload, `0` for CPU-only) |
| `LLM_N_BATCH` | No | `128` | Token batch size per `llama_decode` step |
| `LLM_N_THREADS` | No | `6` | Physical CPU threads for inference |
| `LLM_CONTINUOUS_BATCH_SLOTS` | No | `2` | Number of concurrent sequence slots (`n_seq_max`) sharing the model |
| `USE_MMAP` | No | `true` | Memory-map GGUF weights |
| `USE_MLOCK` | No | `false` | Lock weights in physical RAM |
| `LLM_MIN_AVAILABLE_RAM_MB` | No | `3000` | Minimum free RAM required before loading a chat model |

### Orchestrator, ChromaDB, Web Search & Vision
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ORCH_ENABLED` | No | `true` | Enable RAG, Web Search, and Vision orchestration |
| `ORCH_CHROMA_DIR` | No | `./chroma_db` | Persistent ChromaDB directory path |
| `ORCH_EMBEDDING_MODEL` | No | `BAAI/bge-small-en-v1.5` | Local embedding model (with automatic offline 384-d hash fallback) |
| `ORCH_DDG_MAX_RESULTS` | No | `5` | Maximum search results fetched per query |
| `ORCH_WEB_CACHE_TTL_HOURS` | No | `24` | Cache TTL in hours for `web_cache` chunks |
| `ORCH_SCRAPE_TIMEOUT_SEC` | No | `5.0` | Per-page HTTP scraping timeout in seconds |
| `ORCH_CHUNK_SIZE_CHARS` | No | `900` | Character chunk size for RAG & web ingestion |
| `ORCH_CHUNK_OVERLAP_CHARS` | No | `120` | Overlap between consecutive chunks |
| `ORCH_MAX_RAG_CHUNKS` | No | `3` | Maximum chunks injected into `<context>` per query |
| `ORCH_MAX_DISTANCE` | No | `0.65` | Maximum cosine distance threshold for vector search hits |
| `VISION_MODEL_PATH` | No | `/mnt/llm_models/LLM_Models/Vision_Models` | Dedicated directory containing Vision `.gguf` models and `mmproj` files |
| `VISION_API_URL` | No | - | Optional remote Vision API endpoint (e.g. `http://vision-node:8000/app/v1/vision/describe`) for multi-node clusters |
| `ORCH_VISION_GPU_LAYERS` | No | `0` | GPU layers for Vision model (`0` keeps Vision on CPU so active chat LLM stays in VRAM; `-1` for GPU) |

---

## License

MIT License

## Author

**Talha Khalid**
