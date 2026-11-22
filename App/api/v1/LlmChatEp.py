from App.repository.llmRepository import llmChatRepo
from App.core.orchestrator import ChatOrchestrator, detect_upload_kind
from App.api.dependencies.auth import decode_jwt
from App.core.exceptions import (
    FileTooLargeError,
    ImageTooLargeError,
    InvalidCollectionError,
    InvalidFileError,
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


def _resolve_tenant(
    req: Request,
    explicit_user_id: str | None = None,
    explicit_session_id: str | None = None,
) -> tuple[str, str]:
    """
    Resolve `(user_id, session_id)` for strict multi-tenant ChromaDB & RAM isolation:
      1. JWT `Authorization: Bearer <token>` (`sub` / `user_id`) if present
      2. Explicit `user_id` / `session_id` in request body, form, or query params
      3. `X-User-ID` and `X-Session-ID` HTTP headers
      4. Auto-generated random UUID (`user_<hex>` / `sess_<hex>`) if omitted by client
    """
    jwt_user: str | None = None
    auth_hdr = req.headers.get("authorization", "")
    if auth_hdr.lower().startswith("bearer "):
        token = auth_hdr.split(" ", 1)[1].strip()
        payload = decode_jwt(token) if token else None
        if isinstance(payload, dict):
            jwt_user = str(payload.get("sub") or payload.get("user_id") or "") or None

    clean_uid = explicit_user_id.strip() if (explicit_user_id and explicit_user_id.strip() not in ("", "string", "null", "None")) else None
    clean_sid = explicit_session_id.strip() if (explicit_session_id and explicit_session_id.strip() not in ("", "string", "null", "None")) else None

    uid = (
        jwt_user
        or clean_uid
        or req.headers.get("x-user-id")
        or f"user_{uuid4().hex[:16]}"
    )
    sid = (
        clean_sid
        or req.headers.get("x-session-id")
        or f"sess_{uuid4().hex[:16]}"
    )
    return str(uid).strip(), str(sid).strip()


def _parse_system_prompts_chain(system_prompts_raw: str | None) -> list[dict[str, str]]:
    """
    Parse `system_prompts` form field containing a JSON list of objects (or strings)
    for chaining multiple system instructions before the main user prompt.
    Accepted formats:
      - `[{"content": "You are Dr. Aris..."}, {"content": "Always respond in Markdown."}]`
      - `[{"role": "system", "content": "..."}, {"system": "..."}]`
      - `["System prompt 1", "System prompt 2"]`
      - Single JSON object `{"content": "..."}` or plain string fallback
    """
    if not system_prompts_raw or not system_prompts_raw.strip() or system_prompts_raw.strip() == "string":
        return []
    raw = system_prompts_raw.strip()
    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except Exception:
        # Plain text fallback if user typed a raw string instead of JSON list
        return [{"role": "system", "content": raw}]

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [{"role": "system", "content": str(parsed)}]

    chain: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, str) and item.strip():
            chain.append({"role": "system", "content": item.strip()})
        elif isinstance(item, dict):
            txt = (
                item.get("content")
                or item.get("system")
                or item.get("prompt")
                or item.get("text")
                or ""
            )
            if isinstance(txt, str) and txt.strip() and txt.strip() != "string":
                chain.append({"role": "system", "content": txt.strip()})
    return chain


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
    image: UploadFile | None = File(
        default=None,
        description="Optional file or image upload (auto-detects Images, PDF, Scanned PDF, DOCX, XLSX, CSV, ZIP, JSON, TXT, Code strictly in RAM)",
    ),
    repo: llmChatRepo = Depends(get_repo),
):
    uploaded_bytes: bytes | None = None
    uploaded_filename: str | None = None
    uploaded_kind: str | None = None
    ctype = req.headers.get("content-type", "").lower()

    if ctype.startswith("application/json"):
        body = await req.json()
        request = ChatRequest(**body)
    else:
        if request is None:
            raise MissingRequestFieldError("Missing 'request' JSON form field")
        if image is not None and image.filename:
            uploaded_bytes = await image.read()
            uploaded_filename = image.filename
            if not uploaded_bytes:
                raise InvalidFileError("Uploaded file is empty")
            if len(uploaded_bytes) > MAX_IMAGE_UPLOAD_BYTES:
                raise FileTooLargeError("Uploaded file exceeds 15MB limit")
            uploaded_kind, _ = detect_upload_kind(uploaded_bytes, uploaded_filename)

    active_model = await _require_loaded_model(repo, request.model)
    request.model = active_model
    tenant_user_id, tenant_session_id = _resolve_tenant(
        req,
        explicit_user_id=getattr(request, "user_id", None),
        explicit_session_id=getattr(request, "session_id", None),
    )
    logger.info(
        f"Chat request - Model: {request.model}, User: {tenant_user_id}, Session: {tenant_session_id}, "
        f"Messages: {len(request.messages)}, Upload: {uploaded_filename} ({uploaded_kind})"
    )

    # ============================================
    # 1. ORCHESTRATION (RAG + WEB SEARCH + DIRECT URL SCRAPE + VISION + UNIVERSAL DOC PARSER)
    # ============================================
    orch_ctx = None
    should_orchestrate = settings.ORCH_ENABLED and (
        getattr(request, "use_orchestrator", True)
        or getattr(request, "use_web", None) is True
        or getattr(request, "use_rag", None) is True
        or uploaded_bytes is not None
        or getattr(request, "image_base64", None) is not None
    )
    if should_orchestrate:
        orch_ctx = await _orchestrator.prepare_context(
            request.messages,
            file_bytes=uploaded_bytes,
            filename=uploaded_filename,
            image_base64=getattr(request, "image_base64", None),
            force_web=getattr(request, "use_web", None),
            force_rag=getattr(request, "use_rag", None),
            user_id=tenant_user_id,
            session_id=tenant_session_id,
            model_name=active_model,
        )
        raw_messages = orch_ctx.enriched_messages
        logger.info(
            f"🧭 Orchestrator route={orch_ctx.route.model_dump()} | user={tenant_user_id} | session={tenant_session_id} | "
            f"doc={f'{orch_ctx.document_report.file_type}:{orch_ctx.document_report.doc_profile}' if orch_ctx.document_report else None} | "
            f"citations={len(orch_ctx.citations)} | metrics={orch_ctx.metrics_ms}"
        )
    else:
        from App.core.orchestrator import get_model_base_prompt
        base_persona = get_model_base_prompt(active_model)
        user_sys = [
            m.content for m in request.messages
            if m.role == "system" and isinstance(m.content, str) and m.content.strip()
        ]
        non_sys = [
            {
                "role": m.role,
                "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
            }
            for m in request.messages
            if m.role != "system"
        ]
        sys_body = base_persona
        if user_sys:
            sys_body += "\n\n[User Custom Instructions]:\n" + "\n\n".join(f"- {s}" for s in user_sys)
        raw_messages = [{"role": "system", "content": sys_body}] + non_sys

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
                    "user_id": tenant_user_id,
                    "session_id": tenant_session_id,
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
                    user_id=tenant_user_id,
                    session_id=tenant_session_id,
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

    # If the LLM invoked one of the orchestrator's internal document/web tools, execute it and synthesize final answer
    internal_tools = {
        "search_document",
        "read_document_pages",
        "query_table_data",
        "inspect_page_visual",
        "scrape_website",
        "web_search",
    }
    if tool_calls and tool_calls[0]["function"]["name"] in internal_tools:
        t_call = tool_calls[0]["function"]
        t_name = t_call["name"]
        try:
            t_args = json.loads(t_call["arguments"]) if isinstance(t_call["arguments"], str) else t_call["arguments"]
        except Exception:
            t_args = {}
        logger.info(f"🔧 Executing internal orchestrator tool: {t_name}({t_args}) for user={tenant_user_id}")
        tool_output = await _orchestrator.execute_internal_tool(
            t_name,
            t_args,
            user_id=tenant_user_id,
            session_id=tenant_session_id,
        )
        followup_msgs = list(raw_messages) + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": f"<tool_output name='{t_name}'>\n{tool_output}\n</tool_output>\nNow answer the user's question directly using this tool output."},
        ]
        followup_prompt = await repo.format_messages_native(followup_msgs)
        async with _orchestrator.vram_gate:
            ai_msg2 = await repo.chat_with_metadata(prompt=followup_prompt, **gen_kwargs)
        content = ai_msg2.content if hasattr(ai_msg2, "content") else str(ai_msg2)
        meta2 = getattr(ai_msg2, "response_metadata", {}) or {}
        u2 = meta2.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        exact_usage = {
            "prompt_tokens": exact_usage.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
            "completion_tokens": exact_usage.get("completion_tokens", 0) + u2.get("completion_tokens", 0),
            "total_tokens": exact_usage.get("total_tokens", 0) + u2.get("total_tokens", 0),
        }
        tool_calls = None

    finish_reason = "tool_calls" if tool_calls else meta.get("finish_reason", "stop")

    if orch_ctx is not None:
        await _orchestrator.memory.record_chat_turn(
            query=orch_ctx.user_query,
            answer=content,
            model_used=request.model,
            route=orch_ctx.route,
            user_id=tenant_user_id,
            session_id=tenant_session_id,
        )

    logger.info(f"✅ LLM response received: {content[:100]}...")

    return ChatResponse(
        id=f"chatcmpl-{uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        user_id=tenant_user_id,
        session_id=tenant_session_id,
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
    req: Request,
    collection: str = Query(default="chat_history", description="chat_history | web_cache | image_index | docs_rag"),
    user_id: str | None = Query(default=None, description="Filter by user_id (defaults to current user)"),
    session_id: str | None = Query(default=None, description="Optional session_id filter"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """View stored records from the local ChromaDB vector store isolated by user_id/session_id."""
    uid, _ = _resolve_tenant(req, explicit_user_id=user_id, explicit_session_id=session_id)
    return await orch.memory.list_entries(
        collection_name=collection,
        limit=limit,
        offset=offset,
        user_id=uid if user_id is not None or req.headers.get("authorization") or req.headers.get("x-user-id") else user_id,
        session_id=session_id,
    )


@llmRouter.get("/history/search")
async def search_vector_history(
    req: Request,
    q: str = Query(..., min_length=1, description="Semantic search query"),
    collection: str = Query(default="chat_history", description="chat_history | web_cache | image_index | docs_rag"),
    user_id: str | None = Query(default=None, description="Tenant user_id filter"),
    session_id: str | None = Query(default=None, description="Optional session_id filter"),
    n_results: int = Query(default=5, ge=1, le=20),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Semantic search across ChromaDB history strictly isolated by user_id and optional session_id."""
    if collection not in orch.memory.VALID_COLLECTIONS:
        raise InvalidCollectionError(
            f"Invalid collection '{collection}'. Choose from {orch.memory.VALID_COLLECTIONS}"
        )
    uid, _ = _resolve_tenant(req, explicit_user_id=user_id, explicit_session_id=session_id)
    effective_uid = uid if (user_id is not None or req.headers.get("authorization") or req.headers.get("x-user-id")) else None
    where = None if collection == "web_cache" else orch.memory.build_tenant_where(user_id=effective_uid, session_id=session_id)
    hits = await orch.memory.query(collection_name=collection, query=q, n_results=n_results, where=where)
    return {"query": q, "collection": collection, "user_id": effective_uid, "session_id": session_id, "count": len(hits), "results": hits}


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
    req: Request,
    image: UploadFile = File(..., description="Image file processed strictly in RAM"),
    prompt: str = Form(default="", description="Optional question or focus hint for Qwen2-VL"),
    user_id: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    repo: llmChatRepo = Depends(get_repo),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Upload an image -> decode in RAM with Qwen2-VL-2B -> index in ChromaDB isolated by (user_id, session_id)."""
    await _require_loaded_model(repo)
    uid, sid = _resolve_tenant(req, explicit_user_id=user_id, explicit_session_id=session_id)
    image_bytes = await image.read()
    if not image_bytes:
        raise InvalidImageError("Uploaded image is empty")
    if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
        raise ImageTooLargeError("Image exceeds 15MB limit")

    report = await orch.vision_tool.describe_and_index(
        image_bytes=image_bytes,
        user_hint=prompt,
        user_id=uid,
        session_id=sid,
    )
    return {"status": "ok", "user_id": uid, "session_id": sid, "vision_report": report.model_dump()}


@llmRouter.post("/chat/completions/upload")
async def chat_with_uploaded_image(
    req: Request,
    model: str = Form(...),
    prompt: str = Form(..., description="Main user prompt/question"),
    system_prompts: str | None = Form(
        default='[{"content": "You are a helpful AI assistant."}]',
        description='JSON list of system prompt objects for multi-chain system instructions, e.g. [{"content": "You are an expert..."}, {"content": "Be concise."}]',
    ),
    image: UploadFile | None = File(
        default=None,
        description="Optional file or image processed strictly in RAM (auto-detects Image, PDF, DOCX, XLSX, CSV, ZIP, TXT, Code)",
    ),
    user_id: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    use_web: bool | None = Form(default=None),
    use_rag: bool | None = Form(default=None),
    temperature: float | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    repo: llmChatRepo = Depends(get_repo),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Chat with optional uploaded file/image + chained JSON list of system prompts + main user prompt (strictly isolated by user_id & session_id)."""
    active_model = await _require_loaded_model(repo, model)
    uid, sid = _resolve_tenant(req, explicit_user_id=user_id, explicit_session_id=session_id)

    uploaded_bytes: bytes | None = None
    uploaded_filename: str | None = None
    if image is not None and image.filename:
        uploaded_bytes = await image.read()
        uploaded_filename = image.filename
        if not uploaded_bytes:
            raise InvalidFileError("Uploaded file is empty")
        if len(uploaded_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise FileTooLargeError("Uploaded file exceeds 15MB limit")

    sys_chain = _parse_system_prompts_chain(system_prompts)
    messages_chain: list[dict[str, str]] = []
    if sys_chain:
        # Combine chained system prompts in order into the system role (and keep individual entries if supported)
        combined_sys = "\n\n".join(f"[System Instruction {i+1}]\n{s['content']}" if len(sys_chain) > 1 else s["content"] for i, s in enumerate(sys_chain))
        messages_chain.append({"role": "system", "content": combined_sys})
    messages_chain.append({"role": "user", "content": prompt})

    orch_ctx = await orch.prepare_context(
        messages_chain,
        file_bytes=uploaded_bytes,
        filename=uploaded_filename,
        force_web=use_web,
        force_rag=use_rag,
        user_id=uid,
        session_id=sid,
        model_name=active_model,
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
        user_id=uid,
        session_id=sid,
    )
    return {
        "id": f"chatcmpl-{uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": active_model,
        "user_id": uid,
        "session_id": sid,
        "system_prompts_count": len(sys_chain),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "vision_report": orch_ctx.vision_report.model_dump() if orch_ctx.vision_report else None,
        "document_report": (
            orch_ctx.document_report.model_dump(exclude={"full_text", "pages"})
            if orch_ctx.document_report
            else None
        ),
        "citations": [c.model_dump() for c in orch_ctx.citations],
        "metrics_ms": orch_ctx.metrics_ms,
    }


@llmRouter.post("/chat/completions/code")
async def chat_with_uploaded_code(
    req: Request,
    model: str = Form(...),
    prompt: str = Form(..., description="Main coding question/instruction (user prompt)"),
    system_prompts: str | None = Form(
        default='[{"content": "You are a principal software architect and polyglot compiler/AST engineer."}]',
        description='JSON list of system prompt objects for multi-chain coding instructions, e.g. [{"content": "You are a senior Rust/Python architect."}, {"content": "Focus on time complexity and clean code."}]',
    ),
    code_file: UploadFile | None = File(
        default=None,
        description="Upload ANY code file in ANY programming language (processed 100% in RAM via AST splitting & token-saving skeleton compression)",
    ),
    user_id: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    use_web: bool | None = Form(default=False),
    use_rag: bool | None = Form(default=True),
    temperature: float | None = Form(default=0.2),
    max_tokens: int | None = Form(default=1024),
    repo: llmChatRepo = Depends(get_repo),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """
    Dedicated Coding & AST Analysis Endpoint:
    - Accepts code files in ANY language in the world (Python, JS/TS, C/C++, Rust, Go, Java, C#, PHP, Ruby, Solidity, SQL, Bash, etc.).
    - Splits code into AST nodes in RAM and indexes every function/class/symbol into ChromaDB (`docs_rag`) isolated by `(user_id, session_id)`.
    - Automatically compresses large code files into an AST Skeleton + top relevant AST symbols when code length exceeds the context window budget.
    """
    from App.core.orchestrator import UniversalCodeASTProcessor, ParsedPage, ParsedDocumentReport

    active_model = await _require_loaded_model(repo, model)
    uid, sid = _resolve_tenant(req, explicit_user_id=user_id, explicit_session_id=session_id)

    t0 = time.perf_counter()
    ast_report: dict | None = None
    code_context_block = ""
    vision_report_obj = None
    is_image_code = False

    # Calculate safe token-saving character budget based on active model's per-sequence KV context window
    per_seq_ctx = max(512, repo.n_ctx // max(1, repo.continuous_batch_slots))
    reserve_gen = min(int(max_tokens or 1024), max(256, per_seq_ctx // 3))
    max_code_chars = max(1400, (per_seq_ctx - reserve_gen - 450) * 3)

    if code_file is not None and code_file.filename:
        code_bytes = await code_file.read()
        if not code_bytes:
            raise InvalidFileError("Uploaded code file is empty")
        if len(code_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise FileTooLargeError("Uploaded code file exceeds 15MB limit")

        # Auto-detect if user uploaded an image / screenshot of code instead of raw text
        fname_lower = code_file.filename.lower()
        if (
            fname_lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"))
            or (code_file.content_type and code_file.content_type.startswith("image/"))
            or code_bytes[:8].startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF"))
        ):
            is_image_code = True
            logger.info(f"Code endpoint detected image file '{code_file.filename}'. Running In-RAM Vision OCR...")
            vision_report_obj = await orch.vision_tool.describe(
                image_bytes=code_bytes,
                user_hint=f"Extract and transcribe all code, functions, and text accurately from this image. {prompt}",
                user_id=uid,
                session_id=sid,
            )
            extracted_code = vision_report_obj.visible_text.strip()
            if not extracted_code or len(extracted_code) < 5:
                extracted_code = f"// Code extracted from {code_file.filename}\n// Summary: {vision_report_obj.summary}"
            code_bytes = extracted_code.encode("utf-8")

        ast_report = await asyncio.to_thread(
            UniversalCodeASTProcessor.process_code_in_ram,
            code_bytes,
            code_file.filename,
            prompt,
            max_code_chars,
        )
        code_context_block = ast_report["context_payload"]

        # Index each AST node into ChromaDB `docs_rag` isolated by (uid, sid) so follow-up questions in the same session can query any symbol!
        ast_pages = [
            ParsedPage(
                page_number=idx,
                title=f"{node['kind']}: {node['symbol']} (L{node['start_line']}-L{node['end_line']})",
                text=f"// Symbol: {node['symbol']} | Signature: {node['signature']}\n{node['code']}",
            )
            for idx, node in enumerate(ast_report["ast_nodes"], start=1)
        ] or [ParsedPage(page_number=1, title=code_file.filename, text=code_bytes.decode("utf-8", errors="ignore")[:2400])]

        doc_rep = ParsedDocumentReport(
            doc_id=f"code_{uuid4().hex[:16]}",
            filename=code_file.filename,
            file_type="code",
            doc_profile="structured_code_or_json",
            mime_type="text/plain",
            user_id=uid,
            session_id=sid,
            total_pages=len(ast_pages),
            total_chars=ast_report["raw_chars"],
            total_chunks=len(ast_pages),
            toc_summary=ast_report["skeleton"][:800],
            pages=ast_pages,
            full_text=code_context_block,
        )
        await orch.memory.index_parsed_document(doc_rep)

    sys_chain = _parse_system_prompts_chain(system_prompts)
    if is_image_code:
        sys_chain.insert(0, {
            "content": (
                "The user uploaded an image/screenshot of code. It has ALREADY been transcribed into source code text via Vision OCR in the code context block. "
                "You have full access to the code. You MUST analyze this code, identify and explain all bugs/errors, and provide the corrected code. "
                "DO NOT claim 'I cannot view images' or 'I am a text-based model'."
            )
        })
    messages_chain: list[dict[str, str]] = []
    if sys_chain:
        combined_sys = "\n\n".join(
            f"[System Instruction {i+1}]\n{s['content']}" if len(sys_chain) > 1 else s["content"]
            for i, s in enumerate(sys_chain)
        )
        messages_chain.append({"role": "system", "content": combined_sys})
    else:
        messages_chain.append({
            "role": "system",
            "content": "You are a principal software architect and polyglot AST engineer. Provide accurate, production-ready code and explanations.",
        })

    user_content = f"{code_context_block}\n\nCoding Task / Question: {prompt}".strip() if code_context_block else prompt
    messages_chain.append({"role": "user", "content": user_content})

    # Also allow retrieving previously uploaded code AST chunks or session memory in the same (uid, sid)
    orch_ctx = await orch.prepare_context(
        messages_chain,
        force_web=use_web,
        force_rag=bool(use_rag and code_file is None),
        user_id=uid,
        session_id=sid,
        model_name=active_model,
    )
    formatted_prompt = await repo.format_messages_native(orch_ctx.enriched_messages)
    gen_kwargs = {k: v for k, v in {"temperature": temperature, "max_tokens": max_tokens}.items() if v is not None}
    async with orch.vram_gate:
        ai_msg = await repo.chat_with_metadata(prompt=formatted_prompt, **gen_kwargs)

    content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
    meta = getattr(ai_msg, "response_metadata", {}) or {}
    exact_usage = meta.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    await orch.memory.record_chat_turn(
        query=prompt,
        answer=content,
        model_used=active_model,
        route=orch_ctx.route,
        user_id=uid,
        session_id=sid,
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "id": f"chatcmpl-{uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": active_model,
        "user_id": uid,
        "session_id": sid,
        "system_prompts_count": len(sys_chain),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": meta.get("finish_reason", "stop"),
            }
        ],
        "usage": exact_usage,
        "vision_report": vision_report_obj.model_dump() if vision_report_obj else None,
        "ast_report": (
            {
                "filename": ast_report["filename"],
                "language": ast_report["language"],
                "raw_chars": ast_report["raw_chars"],
                "compact_chars": ast_report["compact_chars"],
                "compressed_via_ast": ast_report["compressed_via_ast"],
                "max_code_chars_budget": max_code_chars,
                "total_ast_nodes": ast_report["total_ast_nodes"],
                "selected_ast_symbols": ast_report["selected_ast_symbols"],
            }
            if ast_report
            else None
        ),
        "citations": [c.model_dump() for c in orch_ctx.citations],
        "metrics_ms": {**orch_ctx.metrics_ms, "total_code_ep_ms": elapsed_ms},
    }