from App.repository.llmRepository import llmChatRepo
from App.core.orchestrator import ChatOrchestrator
from App.core.exceptions import (
    ImageTooLargeError,
    InvalidCollectionError,
    InvalidImageError,
    MissingRequestFieldError,
    ModelNotLoadedError,
)
from fastapi import APIRouter, Request, Depends, Query, File, Form, UploadFile
from App.models.llmModels import ChatChoice, ChatMessage, ChatRequest, ChatResponse
from App.core.settings import settings
from fastapi.responses import StreamingResponse
from uuid import uuid4
import asyncio
import time
import json
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

llmRouter = APIRouter(tags=["LLM Chat Model"])
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,      # Max chars per chunk
    chunk_overlap=200,    # Overlap to maintain context
    separators=[
        "\n\n",          # Split on paragraphs first
        "\n",            # Then new lines
        ". ",            # Then sentences
        ",",             # Then clauses
        " ",             # Then words
        ""               # Finally characters
    ],
    length_function=len,
)
# ============================================
# Repository Instance
# ============================================
_repo = llmChatRepo(
    model_path=settings.LLM_MODEL_PATH,
    n_gpu_layers=settings.LLM_N_GPU_LAYERS,
    n_ctx=settings.LLM_N_CTX,
    n_batch=settings.LLM_N_BATCH,
    n_threads=settings.LLM_N_THREADS,
    verbose=settings.LLM_VERBOSE,
    use_mmap=settings.USE_MMAP,
    use_mlock=settings.USE_MLOCK,
    continuous_batch_slots=settings.LLM_CONTINUOUS_BATCH_SLOTS,
    available_ram=settings.LLM_MIN_AVAILABLE_RAM_MB
)

def get_repo() -> llmChatRepo:
    """Dependency to get the repository instance."""
    return _repo

_orchestrator = ChatOrchestrator(llm=_repo)

def get_orchestrator() -> ChatOrchestrator:
    """Dependency to get the orchestrator instance."""
    return _orchestrator

def split_large_messages(messages, splitter, max_chunk_size: int = 4000) -> str:
    """
    Build prompt from messages with automatic splitting of large content.
    Supports both ChatMessage objects and plain dicts.
    """
    prompt = ""
    
    for msg in messages:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        raw_content = msg["content"] if isinstance(msg, dict) else msg.content
        content = raw_content if isinstance(raw_content, str) else json.dumps(raw_content)
        
        # Check if content needs splitting
        if len(content) > max_chunk_size:
            logger.info(f"📝 Splitting {role} message: {len(content)} chars -> chunks")
            chunks = splitter.split_text(content)
            
            # Add each chunk with part numbering
            for i, chunk in enumerate(chunks):
                part_label = f" (Part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
                
                if role == "system":
                    prompt += f"[SYSTEM]{part_label} {chunk}\n"
                elif role == "user":
                    prompt += f"[USER]{part_label} {chunk}\n"
                elif role == "assistant":
                    prompt += f"[ASSISTANT]{part_label} {chunk}\n"
            
            logger.info(f"✅ Split into {len(chunks)} chunks")
        else:
            # Normal content - no splitting needed
            if role == "system":
                prompt += f"[SYSTEM] {content}\n"
            elif role == "user":
                prompt += f"[USER] {content}\n"
            elif role == "assistant":
                prompt += f"[ASSISTANT] {content}\n"
    
    prompt += "[ASSISTANT] "
    return prompt

def _inject_tools_into_messages(
    messages: list[dict],
    tools: list[dict] | None,
    tool_choice: str | dict | None,
) -> list[dict]:
    """Inject OpenAI tool definitions cleanly if tools are requested and tool_choice != 'none'."""
    if not tools or tool_choice == "none":
        return messages
    tool_defs = json.dumps(tools, indent=2)
    tool_instruction = (
        "You have access to the following tools/functions. "
        "If you decide to call a tool, respond ONLY with a JSON object in this exact format:\n"
        '{"name": "<tool_name>", "arguments": {<args>}}\n\n'
        f"Available tools:\n{tool_defs}"
    )
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{out[0]['content']}\n\n{tool_instruction}".strip()
    else:
        out.insert(0, {"role": "system", "content": tool_instruction})
    return out


def _extract_tool_calls(content: str, tools: list[dict] | None) -> list[dict] | None:
    """Parse JSON tool call output into OpenAI `tool_calls` format when valid."""
    if not tools or not content:
        return None
    tool_names = {
        (t.get("function", {}) if isinstance(t, dict) else {}).get("name") or t.get("name")
        for t in tools
    }
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict) and obj.get("name") in tool_names:
        args = obj.get("arguments", obj.get("parameters", {}))
        return [{
            "id": f"call_{uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": obj["name"],
                "arguments": args if isinstance(args, str) else json.dumps(args),
            },
        }]
    return None


async def _require_loaded_model(repo: llmChatRepo, requested_model: str | None = None) -> str:
    """Ensure a model is already loaded (never auto-load on chat/vision endpoints)."""
    is_ready = await repo.is_ready()
    loaded_models = await repo.get_loaded_models_list()
    if not is_ready or not loaded_models:
        raise ModelNotLoadedError("No model is currently loaded. Please load a model first via POST /app/v1/load_model.")
    active = loaded_models[0]
    if requested_model and requested_model not in ("string", active):
        raise ModelNotLoadedError(
            f"Requested model '{requested_model}' is not loaded (active model: '{active}'). "
            "Load it first via POST /app/v1/load_model."
        )
    return active


@llmRouter.post("/chat/completions")
async def completion(
    req: Request,
    request: ChatRequest | None = Form(default=None, description="ChatRequest JSON"),
    image: UploadFile | None = File(default=None, description="Optional image upload (processed strictly in RAM)"),
    repo: llmChatRepo = Depends(get_repo),
):
    image_bytes: bytes | None = None
    ctype = req.headers.get("content-type", "").lower()

    if ctype.startswith("application/json"):
        body = await req.json()
        request = ChatRequest(**body)
    else:
        if request is None:
            raise MissingRequestFieldError("Missing 'request' JSON form field")
        if image is not None and image.filename:
            image_bytes = await image.read()
            if not image_bytes:
                raise InvalidImageError("Uploaded image is empty")
            if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
                raise ImageTooLargeError("Image exceeds 15MB limit")

    active_model = await _require_loaded_model(repo, request.model)
    request.model = active_model
    logger.info(f"Chat request - Model: {request.model}, Messages: {len(request.messages)}, Image: {image_bytes is not None}")

    # ============================================
    # 1. ORCHESTRATION (RAG + WEB SEARCH + VISION) -> NATIVE TEMPLATE PROMPT
    # ============================================
    orch_ctx = None
    should_orchestrate = settings.ORCH_ENABLED and (
        getattr(request, "use_orchestrator", True)
        or getattr(request, "use_web", None) is True
        or getattr(request, "use_rag", None) is True
        or image_bytes is not None
        or getattr(request, "image_base64", None) is not None
    )
    if should_orchestrate:
        orch_ctx = await _orchestrator.prepare_context(
            request.messages,
            image_bytes=image_bytes,
            image_base64=getattr(request, "image_base64", None),
            force_web=getattr(request, "use_web", None),
            force_rag=getattr(request, "use_rag", None),
        )
        raw_messages = orch_ctx.enriched_messages
        logger.info(
            f"🧭 Orchestrator route={orch_ctx.route.model_dump()} | "
            f"citations={len(orch_ctx.citations)} | metrics={orch_ctx.metrics_ms}"
        )
    else:
        raw_messages = [
            {
                "role": m.role,
                "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
            }
            for m in request.messages
        ]

    if not raw_messages:
        raw_messages = [{"role": "user", "content": "Hello"}]

    raw_messages = _inject_tools_into_messages(raw_messages, request.tools, request.tool_choice)
    prompt = await repo.format_messages_native(raw_messages, tools=request.tools)

    logger.info(f"📝 Prompt: {len(prompt)} chars")
    logger.info(f"📄 Preview: {prompt[:150]}...")

    # ============================================
    # 2. CALL THE LOADED LLM (NO AUTO-LOAD)
    # ============================================
    gen_kwargs = {}
    if getattr(request, 'temperature', None) is not None:
        gen_kwargs["temperature"] = request.temperature
    if getattr(request, 'max_tokens', None) is not None:
        gen_kwargs["max_tokens"] = request.max_tokens
    if getattr(request, 'top_p', None) is not None:
        gen_kwargs["top_p"] = request.top_p
    if getattr(request, 'top_k', None) is not None:
        gen_kwargs["top_k"] = request.top_k
    if getattr(request, 'stop', None) is not None:
        gen_kwargs["stop"] = request.stop
    if getattr(request, 'repeat_penalty', None) is not None:
        gen_kwargs["repeat_penalty"] = request.repeat_penalty
    if getattr(request, 'response_format', None) is not None:
        gen_kwargs["response_format"] = request.response_format

    use_stream = request.stream if request.stream is not None else settings.STREAM_RESPONSE
    if use_stream:
        response_id = f"chatcmpl-{uuid4().hex[:8]}"
        stream_task_id = f"strm-{uuid4().hex[:10]}"
        created = int(time.time())
        include_usage = bool(request.stream_options and request.stream_options.get("include_usage"))

        async def stream_response():
            def chunk(delta, finish_reason=None, usage=None):
                payload = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }],
                }
                if usage is not None:
                    payload["usage"] = usage
                return "data: " + json.dumps(payload) + "\n\n"

            collected: list[str] = []
            completed_normally = False
            yield chunk({"role": "assistant"})
            try:
                async with _orchestrator.vram_gate:
                    async for token in repo.stream_llm(prompt, chat_id=stream_task_id, **gen_kwargs):
                        if await req.is_disconnected():
                            logger.info(f"Client disconnected mid-stream; cancelling {stream_task_id}")
                            await repo.cancel_request(stream_task_id)
                            return
                        collected.append(token)
                        yield chunk({"content": token})
                completed_normally = True
                yield chunk({}, finish_reason="stop")
                if include_usage:
                    yield chunk({}, finish_reason=None, usage=repo.pop_last_usage(stream_task_id))
                else:
                    repo.pop_last_usage(stream_task_id)
                yield "data: [DONE]\n\n"
            finally:
                if not completed_normally:
                    await repo.cancel_request(stream_task_id)
                    repo.pop_last_usage(stream_task_id)

            if orch_ctx is not None:
                await _orchestrator.memory.record_chat_turn(
                    query=orch_ctx.user_query,
                    answer="".join(collected),
                    model_used=request.model,
                    route=orch_ctx.route,
                )

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async with _orchestrator.vram_gate:
        ai_msg = await repo.chat_with_metadata(
            prompt=prompt,
            **gen_kwargs
        )

    content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
    meta = getattr(ai_msg, "response_metadata", {}) or {}
    exact_usage = meta.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    tool_calls = _extract_tool_calls(content, request.tools)
    finish_reason = "tool_calls" if tool_calls else meta.get("finish_reason", "stop")

    if orch_ctx is not None:
        await _orchestrator.memory.record_chat_turn(
            query=orch_ctx.user_query,
            answer=content,
            model_used=request.model,
            route=orch_ctx.route,
        )

    logger.info(f"✅ LLM response received: {content[:100]}...")

    return ChatResponse(
        id=f"chatcmpl-{uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=None if tool_calls else content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=exact_usage,
    )


# @llmRouter.post("/chat/completions")
# async def completion(request: ChatRequest, req: Request, repo: llmChatRepo = Depends(get_repo)):
#     try:
#         logger.info(f"Chat request - Model: {request.model}, Messages: {len(request.messages)}")
        
#         # ============================================
#         # 1. CHECK IF REPOSITORY IS READY
#         # ============================================
#         if not await repo.is_ready():
#             logger.warning("Repository not ready - attempting to initialize...")
#             try:
#                 await repo.initialize(model_name=request.model)
#                 logger.info(f"Repository initialized with model: {request.model}")
#             except Exception as e:
#                 logger.error(f"Failed to initialize repository: {e}")
#                 return ChatResponse(
#                     id=f"chatcmpl-{uuid4().hex[:8]}",
#                     object="chat.completion",
#                     created=int(time.time()),
#                     model=request.model,
#                     choices=[
#                         ChatChoice(
#                             index=0,
#                             message=ChatMessage(
#                                 role="assistant",
#                                 content=f"Error: Model '{request.model}' not loaded. Error: {str(e)}"
#                             ),
#                             finish_reason="error"
#                         )
#                     ],
#                     usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
#                 )
        
#         # ============================================
#         # 2. BUILD PROMPT FROM MESSAGES
#         # ============================================
#         prompt = ""
#         for msg in request.messages:
#             if msg.role == "system":
#                 prompt += f"[SYSTEM] {msg.content}\n"
#             elif msg.role == "user":
#                 prompt += f"[USER] {msg.content}\n"
#             elif msg.role == "assistant":
#                 prompt += f"[ASSISTANT] {msg.content}\n"
        
#         prompt += "[ASSISTANT] "
        
#         if not prompt or prompt == "[ASSISTANT] ":
#             prompt = "[USER] Hello\n[ASSISTANT] "
        
#         # ============================================
#         # 3. CALL THE LLM with safe parameter handling
#         # ============================================
#         logger.info(f"Sending prompt to LLM: {prompt[:100]}...")
        
#         # Build kwargs safely
#         gen_kwargs = {
#             "temperature": request.temperature,
#             "max_tokens": request.max_tokens,
#         }
        
#         # Only add optional params if they exist
#         if hasattr(request, 'top_p') and request.top_p is not None:
#             gen_kwargs["top_p"] = request.top_p
        
#         if hasattr(request, 'top_k') and request.top_k is not None:
#             gen_kwargs["top_k"] = request.top_k
        
#         if hasattr(request, 'stop') and request.stop is not None:
#             gen_kwargs["stop"] = request.stop
        
#         response = await repo.chat(
#             prompt=prompt,
#             **gen_kwargs
#         )
        
#         content = response if isinstance(response, str) else str(response)
        
#         logger.info(f"LLM response received: {content[:100]}...")
        
#         return ChatResponse(
#             id=f"chatcmpl-{uuid4().hex[:8]}",
#             object="chat.completion",
#             created=int(time.time()),
#             model=request.model,
#             choices=[
#                 ChatChoice(
#                     index=0,
#                     message=ChatMessage(
#                         role="assistant",
#                         content=content
#                     ),
#                     finish_reason="stop"
#                 )
#             ],
#             usage={
#                 "prompt_tokens": len(prompt.split()),
#                 "completion_tokens": len(content.split()),
#                 "total_tokens": len(prompt.split()) + len(content.split())
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"Error in chat completion: {e}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error occurred during chat with LLM: {str(e)}"
#         )

# ============================================
# LOAD MODEL ENDPOINT
# ============================================

@llmRouter.post("/load_model")
async def load_model(model_name: str, repo: llmChatRepo = Depends(get_repo)):
    """
    Load a specific model into the repository.
    If the model is already loaded, just return a message saying so.
    """
    is_ready = await repo.is_ready()
    loaded_models = await repo.get_loaded_models_list()

    if is_ready and model_name in loaded_models:
        logger.info(f"Model '{model_name}' is already loaded")
        return {
            "status": "already_loaded",
            "message": f"Model '{model_name}' is already loaded and ready to use",
            "model": model_name,
            "ready": True,
            "loaded_models": loaded_models,
        }

    if is_ready and loaded_models:
        current_model = loaded_models[0]
        logger.info(f"Unloading current model '{current_model}' to load '{model_name}'")
        await repo.shutdown()
        logger.info(f"Model '{current_model}' unloaded successfully")

    logger.info(f"Loading model '{model_name}'...")
    await repo.initialize(model_name=model_name)

    return {
        "status": "success",
        "message": f"Model '{model_name}' loaded successfully",
        "model": model_name,
        "ready": await repo.is_ready(),
        "loaded_models": await repo.get_loaded_models_list(),
    }


# ============================================
# UNLOAD MODEL ENDPOINT
# ============================================

@llmRouter.post("/unload_model")
async def unload_model(repo: llmChatRepo = Depends(get_repo)):
    """
    Unload the currently loaded model.
    If no model is loaded, just return a message saying so.
    """
    is_ready = await repo.is_ready()
    loaded_models = await repo.get_loaded_models_list()

    if not is_ready or not loaded_models:
        logger.info("No model is currently loaded")
        return {
            "status": "already_unloaded",
            "message": "ℹ️ No model is currently loaded. Nothing to unload.",
            "ready": False,
            "loaded_models": [],
        }

    current_model = loaded_models[0]
    logger.info(f"Unloading model '{current_model}'...")
    await repo.shutdown()
    logger.info(f"Model '{current_model}' unloaded successfully")

    return {
        "status": "success",
        "message": f"✅ Model '{current_model}' unloaded successfully",
        "unloaded_model": current_model,
        "ready": await repo.is_ready(),
        "loaded_models": await repo.get_loaded_models_list(),
    }


# ============================================
# HEALTH CHECK ENDPOINT
# ============================================

@llmRouter.get("/health")
async def health_check(repo: llmChatRepo = Depends(get_repo)):
    """Check if the LLM service is ready."""
    is_ready = await repo.is_ready()
    loaded_models = await repo.get_loaded_models_list()
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "ready": is_ready,
        "models_loaded": len(loaded_models),
        "loaded_models": loaded_models,
        "message": "LLM service is ready" if is_ready else "No model loaded. Call /load_model first.",
    }


# ============================================
# MODELS LIST ENDPOINT
# ============================================

@llmRouter.get("/models")
async def get_dir_models(repo: llmChatRepo = Depends(get_repo)):
    """Get list of available models from the directory."""
    models = await repo.get_models_cdir()
    return {
        "object": "list",
        "data": [
            {
                "id": model["filename"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "path": model["path"],
            }
            for model in models
        ],
    }


# ============================================
# ACTIVE MODEL ENDPOINT
# ============================================

@llmRouter.get("/active_model")
async def get_active_model(repo: llmChatRepo = Depends(get_repo)):
    """View the currently active/loaded model and runtime hardware configuration."""
    is_ready = await repo.is_ready()
    loaded_models = await repo.get_loaded_models_list()
    return {
        "ready": is_ready,
        "active_model": loaded_models[0] if loaded_models else None,
        "loaded_models": loaded_models,
        "orchestrator_enabled": settings.ORCH_ENABLED,
        "vision_model": settings.ORCH_VISION_MODEL,
        "config": {
            "n_ctx": repo.n_ctx,
            "n_gpu_layers": repo.n_gpu_layers,
            "n_batch": repo.n_batch,
            "n_threads": repo.n_threads,
            "continuous_batch_slots": repo.continuous_batch_slots,
        },
    }


# ============================================
# VECTOR DB HISTORY & SEARCH ENDPOINTS
# ============================================

@llmRouter.get("/history")
async def view_vector_history(
    collection: str = Query(default="chat_history", description="chat_history | web_cache | image_index | docs_rag"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """View stored records from the local ChromaDB vector store."""
    return await orch.memory.list_entries(collection_name=collection, limit=limit, offset=offset)


@llmRouter.get("/history/search")
async def search_vector_history(
    q: str = Query(..., min_length=1, description="Semantic search query"),
    collection: str = Query(default="chat_history", description="chat_history | web_cache | image_index | docs_rag"),
    n_results: int = Query(default=5, ge=1, le=20),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Semantic search across ChromaDB history (chat turns, cached web pages, indexed images, docs)."""
    if collection not in orch.memory.VALID_COLLECTIONS:
        raise InvalidCollectionError(
            f"Invalid collection '{collection}'. Choose from {orch.memory.VALID_COLLECTIONS}"
        )
    hits = await orch.memory.query(collection_name=collection, query=q, n_results=n_results)
    return {"query": q, "collection": collection, "count": len(hits), "results": hits}


# ============================================
# WEB SEARCH TEST ENDPOINT
# ============================================

@llmRouter.get("/web_search/test")
async def test_web_search(
    q: str = Query(..., min_length=2, description="Test query for DuckDuckGo + ChromaDB ingestion"),
    bypass_cache: bool = Query(default=True, description="Set true to test live DuckDuckGo search"),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Test DuckDuckGo web search, safe scraping, and ChromaDB ingestion."""
    t0 = time.perf_counter()
    results = await orch.web_tool.search_and_ingest(q, bypass_cache=bypass_cache)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "status": "ok" if results else "no_results",
        "query": q,
        "bypass_cache": bypass_cache,
        "elapsed_ms": elapsed_ms,
        "count": len(results),
        "results": results,
    }


# ============================================
# IN-RAM IMAGE UPLOAD ENDPOINTS (NO DISK SAVE)
# ============================================

MAX_IMAGE_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB RAM guard


@llmRouter.post("/vision/describe")
async def describe_uploaded_image(
    image: UploadFile = File(..., description="Image file processed strictly in RAM"),
    prompt: str = Form(default="", description="Optional question or focus hint for Qwen2-VL"),
    repo: llmChatRepo = Depends(get_repo),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Upload an image -> decode in RAM with Qwen2-VL-2B -> index in ChromaDB (requires loaded model)."""
    await _require_loaded_model(repo)
    image_bytes = await image.read()
    if not image_bytes:
        raise InvalidImageError("Uploaded image is empty")
    if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
        raise ImageTooLargeError("Image exceeds 15MB limit")

    report = await orch.vision_tool.describe_and_index(image_bytes=image_bytes, user_hint=prompt)
    return {"status": "ok", "vision_report": report.model_dump()}


@llmRouter.post("/chat/completions/upload")
async def chat_with_uploaded_image(
    model: str = Form(...),
    prompt: str = Form(...),
    image: UploadFile | None = File(default=None, description="Optional image processed strictly in RAM"),
    use_web: bool | None = Form(default=None),
    use_rag: bool | None = Form(default=None),
    temperature: float | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    repo: llmChatRepo = Depends(get_repo),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Chat with optional uploaded image + prompt (never auto-loads model; raises ModelNotLoadedError if unloaded)."""
    active_model = await _require_loaded_model(repo, model)

    image_bytes: bytes | None = None
    if image is not None and image.filename:
        image_bytes = await image.read()
        if not image_bytes:
            raise InvalidImageError("Uploaded image is empty")
        if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise ImageTooLargeError("Image exceeds 15MB limit")

    orch_ctx = await orch.prepare_context(
        [{"role": "user", "content": prompt}],
        image_bytes=image_bytes,
        force_web=use_web,
        force_rag=use_rag,
    )
    formatted_prompt = await repo.format_messages_native(orch_ctx.enriched_messages)
    gen_kwargs = {k: v for k, v in {"temperature": temperature, "max_tokens": max_tokens}.items() if v is not None}
    async with orch.vram_gate:
        response = await repo.chat(prompt=formatted_prompt, **gen_kwargs)
    content = response if isinstance(response, str) else str(response)

    await orch.memory.record_chat_turn(
        query=orch_ctx.user_query,
        answer=content,
        model_used=active_model,
        route=orch_ctx.route,
    )
    return {
        "id": f"chatcmpl-{uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": active_model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "vision_report": orch_ctx.vision_report.model_dump() if orch_ctx.vision_report else None,
        "citations": [c.model_dump() for c in orch_ctx.citations],
        "metrics_ms": orch_ctx.metrics_ms,
    }