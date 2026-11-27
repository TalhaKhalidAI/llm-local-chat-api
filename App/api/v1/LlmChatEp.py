from typing import Any
from App.services.llm_service import LlmService, get_llm_service, llm_service
from App.repository.llmRepository import llmChatRepo
from App.core.orchestrator import (
    ChatOrchestrator,
    detect_upload_kind,
    strip_ungrounded_citations,
)
from App.api.dependencies.auth import decode_jwt
from App.core.exceptions import (
    FileTooLargeError,
    ImageTooLargeError,
    InvalidCollectionError,
    InvalidFileError,
    InvalidImageError,
    MissingRequestFieldError,
    ModelNotLoadedError,
    VisionModelLoadError,
    VisionModelNotFoundError,
    VisionModelNotLoadedError,
)
from fastapi import APIRouter, Request, Depends, Query, File, Form, UploadFile
from App.models.L import ChatChoice, ChatMessage, ChatRequest, ChatResponse
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
# Repository / Service / Orchestrator Singletons
# ============================================
# Share the exact same underlying llm_repo engine across LlmService, ChatOrchestrator, and VisionDescriberTool
_repo = llm_service.repo

def get_repo() -> LlmService:
    """Dependency to get the LLM Service instance."""
    return llm_service

_orchestrator = ChatOrchestrator(llm=llm_service.repo)

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
    forced_hint = ""
    if tool_choice == "required":
        forced_hint = "You MUST call one of the available tools and output ONLY the JSON object.\n"
    elif isinstance(tool_choice, dict):
        fn_name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        if fn_name:
            forced_hint = f"You MUST call the tool '{fn_name}' and output ONLY the JSON object.\n"

    tool_instruction = (
        "[TOOL CALLING INSTRUCTIONS]:\n"
        "You have access to external tools/functions listed below. "
        "Whenever a tool can fulfill the user's request (such as fetching live data, weather, calculations, or actions), "
        "do NOT guess or fabricate the answer in prose — respond ONLY with a raw JSON object in this exact format:\n"
        '{"name": "<tool_name>", "arguments": {<key>: <value>}}\n'
        f"{forced_hint}\n"
        f"Available tools:\n{tool_defs}"
    )
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{out[0]['content']}\n\n{tool_instruction}".strip()
    else:
        out.insert(0, {"role": "system", "content": tool_instruction})
    return out


def _extract_tool_calls(content: str, tools: list[dict] | None) -> list[dict] | None:
    """Parse JSON tool call output (even if wrapped in markdown fences, <tool_call> tags, or brief prose) into OpenAI `tool_calls` format."""
    if not tools or not content:
        return None
    tool_names = {
        (t.get("function", {}) if isinstance(t, dict) else {}).get("name") or t.get("name")
        for t in tools
        if isinstance(t, dict)
    }
    tool_names.discard(None)
    if not tool_names:
        return None

    candidates: list[str] = [content.strip()]
    import re as _re
    for m in _re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=_re.DOTALL | _re.IGNORECASE):
        candidates.append(m.group(1).strip())
    for m in _re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, flags=_re.DOTALL | _re.IGNORECASE):
        candidates.append(m.group(1).strip())

    decoder = json.JSONDecoder()
    for idx_pos, ch in enumerate(content):
        if ch == "{":
            try:
                parsed_obj, _ = decoder.raw_decode(content[idx_pos:])
                if isinstance(parsed_obj, dict):
                    candidates.append(json.dumps(parsed_obj))
            except Exception:
                continue

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            fn_block = obj.get("function") if isinstance(obj.get("function"), dict) else obj
            name = fn_block.get("name")
            if name in tool_names:
                args = fn_block.get("arguments", fn_block.get("parameters", {}))
                return [{
                    "id": f"call_{uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": name,
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


async def _require_loaded_model(service: LlmService, requested_model: str | None = None) -> str:
    """Ensure a model is already loaded (never auto-load on chat/vision endpoints)."""
    is_ready = await service.is_ready()
    loaded_models = await service.get_loaded_models_list()
    if not is_ready or not loaded_models:
        raise ModelNotLoadedError("No model is currently loaded. Please load a model first via POST /app/v1/load_model.")
    active = loaded_models[0]
    if requested_model and requested_model not in ("string", active):
        raise ModelNotLoadedError(
            f"Requested model '{requested_model}' is not loaded (active model: '{active}'). "
            "Load it first via POST /app/v1/load_model."
        )
    return active


def _require_loaded_vision_model(orch: ChatOrchestrator) -> None:
    """Ensure a vision model is resident in memory or VISION_API_URL is configured in settings/.env."""
    effective_url = (settings.VISION_API_URL or "").strip() or None
    if not orch.vision_tool.is_loaded() and not effective_url:
        raise VisionModelNotLoadedError(
            "Vision model is not loaded locally and no VISION_API_URL was configured in .env. "
            "Please load a vision model via POST /app/v1/vision/load_model or set VISION_API_URL in .env."
        )


_WEATHER_INTENT_RE = None


def _has_weather_intent(text: str) -> bool:
    import re as _re
    return bool(_re.search(r"\b(weather|temperature|forecast|humidity|rain|snow|wind|celsius|fahrenheit|climate|mausam)\b", text or "", _re.IGNORECASE))


def _merge_agentic_and_client_tools(
    orch: ChatOrchestrator,
    client_tools: list[dict] | None,
    allow_web: bool = True,
    allow_rag: bool = True,
    has_local_images: bool = False,
    allow_weather: bool = False,
) -> list[dict]:
    """Combine built-in orchestrator agentic tools with any external client-supplied OpenAI tools without duplicates."""
    vision_ok = orch.vision_tool.is_loaded() or bool((settings.VISION_API_URL or "").strip())
    internal = orch.get_agentic_tools(
        has_local_docs=allow_rag,
        allow_web=allow_web,
        allow_vision=vision_ok,
        has_local_images=has_local_images,
        allow_weather=allow_weather,
    )
    if not client_tools:
        return internal
    existing_names = {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in client_tools
        if isinstance(t, dict)
    }
    merged = list(client_tools)
    for it in internal:
        if it["function"]["name"] not in existing_names:
            merged.append(it)
    return merged


def _map_compact_react_arg(tool_name: str, raw_arg: str) -> dict[str, Any]:
    """Convert compact GBNF ReAct `query_or_arg` string into structured tool arguments (unwrapping nested JSON if needed)."""
    import re as _re
    arg = (raw_arg or "").strip()
    # Unwrap accidental JSON object strings emitted by small models, e.g. '{"location": "London"}'
    if arg.startswith("{") and arg.endswith("}"):
        try:
            parsed = json.loads(arg)
            if isinstance(parsed, dict) and parsed:
                for k in ("url", "location", "city", "query", "filter", "arg"):
                    if k in parsed and isinstance(parsed[k], str) and parsed[k].strip():
                        arg = parsed[k].strip()
                        break
                else:
                    first_str = next((str(v).strip() for v in parsed.values() if isinstance(v, (str, int, float))), "")
                    if first_str:
                        arg = first_str
        except Exception:
            pass
    if tool_name == "search_document":
        return {"query": arg or "overview", "top_k": 3}
    if tool_name == "search_images":
        return {"query": arg or "image"}
    if tool_name == "get_weather":
        clean_city = _re.sub(r"^(?:weather\s+(?:in|for|at)\s+|live\s+weather\s+(?:in|for)\s+)", "", arg, flags=_re.IGNORECASE).strip(" ?!.,'\"")
        return {"location": clean_city or arg}
    if tool_name == "web_search":
        return {"query": arg}
    if tool_name == "research_person":
        return {"query": arg}
    if tool_name == "scrape_website":
        m_u = _re.search(r"https?://[^\s<>'\"\)\]]+", arg)
        return {"url": m_u.group(0).rstrip(".,;:") if m_u else arg}
    if tool_name == "query_table_data":
        return {"filter": arg}
    if tool_name == "read_document_pages":
        nums = [int(x) for x in _re.findall(r"\d+", arg)]
        if len(nums) >= 2:
            return {"start_page": nums[0], "end_page": nums[1]}
        if len(nums) == 1:
            return {"start_page": nums[0], "end_page": nums[0]}
        return {"start_page": 1, "end_page": 2}
    if tool_name == "inspect_page_visual":
        m = _re.match(r"^\s*(\d+)\s*[:\-,\s]\s*(.*)$", arg)
        if m:
            return {"page_num": int(m.group(1)), "question": m.group(2).strip() or "Describe this page."}
        return {"page_num": 1, "question": arg or "Describe this page."}
    return {"query": arg}


async def _execute_hybrid_react_loop(
    *,
    service: LlmService,
    orch: ChatOrchestrator,
    orch_ctx: Any,
    base_messages: list[dict],
    active_tools: list[dict] | None,
    tool_choice: Any,
    gen_kwargs: dict[str, Any],
    user_id: str,
    session_id: str,
    max_steps: int = 3,
    is_agentic: bool = False,
    client_tools: list[dict] | None = None,
    prepare_for_final_stream: bool = False,
) -> dict[str, Any]:
    """
    Hybrid Tool & Multi-Step Autonomous ReAct Engine:
      1. When `is_agentic=True` (and no custom external client tools override):
         Uses a C++ GBNF Grammar-Constrained ReAct Planner (`LlamaGrammar.from_json_schema`)
         with a compact ~90-token tool menu so even quantized 7B/16B GGUF models (`Q4_K_M`)
         NEVER stall in chatty prose ("Let's start with step 1") and execute each reasoning hop
         in ~2 seconds!
      2. When external `client_tools` are provided:
         Supports arbitrary OpenAI `tools` JSON schemas and returns `finish_reason="tool_calls"`.
    """
    from App.core.orchestrator import IntentRouter

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    agent_trace: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    steps_limit = max(1, min(int(max_steps or 3), 6))

    has_external_client_tools = bool(
        client_tools
        and any(
            ((t.get("function") or {}).get("name") or t.get("name")) not in orch.INTERNAL_TOOL_NAMES
            for t in client_tools
            if isinstance(t, dict)
        )
    )

    # =========================================================================
    # PATH A: External OpenAI Function-Calling Tools (e.g., convert_currency)
    # =========================================================================
    if has_external_client_tools or not is_agentic:
        working_msgs = [dict(m) for m in base_messages]
        msgs_with_tools = _inject_tools_into_messages(working_msgs, active_tools, tool_choice)
        step_prompt = await service.format_messages_native(msgs_with_tools, tools=active_tools)
        async with orch.vram_gate:
            ai_msg = await service.chat_with_metadata(prompt=step_prompt, **gen_kwargs)
        step_content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
        meta = getattr(ai_msg, "response_metadata", {}) or {}
        u = meta.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        total_usage = dict(u)

        parsed_calls = _extract_tool_calls(step_content, active_tools)
        if not parsed_calls:
            return {
                "stream_messages": None,
                "content": step_content,
                "tool_calls": None,
                "finish_reason": meta.get("finish_reason", "stop"),
                "usage": total_usage,
                "agent_trace": agent_trace,
            }
        t_name = parsed_calls[0]["function"]["name"]
        if t_name not in orch.INTERNAL_TOOL_NAMES:
            return {
                "stream_messages": None,
                "content": "",
                "tool_calls": parsed_calls,
                "finish_reason": "tool_calls",
                "usage": total_usage,
                "agent_trace": agent_trace,
            }

    # =========================================================================
    # PATH B: True Multi-Step GBNF Grammar-Constrained ReAct Agent (`is_agentic=True`)
    # =========================================================================
    import re as _re
    from App.core.orchestrator import extract_urls_from_text

    user_q = orch_ctx.user_query if orch_ctx is not None else next((m.get("content", "") for m in reversed(base_messages) if m.get("role") == "user"), "")

    # Extract <conversation_memory> & remembered real URLs so the ReAct Planner is NEVER blind to prior session turns!
    planner_memory_blocks: list[str] = []
    remembered_urls: list[str] = []
    for bm in base_messages:
        c_str = str(bm.get("content") or "")
        for mem_m in _re.finditer(r"<conversation_memory>([\s\S]*?)</conversation_memory>", c_str):
            mem_txt = mem_m.group(1).strip()
            if mem_txt:
                planner_memory_blocks.append(mem_txt[:700])
                remembered_urls.extend(extract_urls_from_text(mem_txt))
        if bm.get("role") == "user":
            remembered_urls.extend(extract_urls_from_text(c_str))
    remembered_urls = [u for u in dict.fromkeys(remembered_urls) if not orch.web_tool._is_placeholder_url(u)][:5]

    memory_section = ""
    if planner_memory_blocks or remembered_urls:
        mem_parts: list[str] = []
        if planner_memory_blocks:
            mem_parts.append("\n".join(planner_memory_blocks)[:1150])
        if remembered_urls:
            mem_parts.append("Remembered Real URLs in this session: " + ", ".join(remembered_urls))
        memory_section = "\n\n[Session Conversation Memory]:\n" + "\n".join(mem_parts)

    internal_tool_names = [
        (t.get("function") or {}).get("name")
        for t in (active_tools or [])
        if isinstance(t, dict) and (t.get("function") or {}).get("name") in orch.INTERNAL_TOOL_NAMES
    ]
    if not internal_tool_names:
        internal_tool_names = ["web_search", "scrape_website", "research_person"]
        if _has_weather_intent(user_q):
            internal_tool_names.append("get_weather")
    elif not _has_weather_intent(user_q) and "get_weather" in internal_tool_names:
        # Prevent unrelated small-model weather hallucinations when user never asked about weather
        internal_tool_names = [n for n in internal_tool_names if n != "get_weather"]

    _full_tool_catalog = {
        "search_document": "- search_document: Search user's uploaded/local session documents in ChromaDB (`query_or_arg` = search keywords)",
        "read_document_pages": "- read_document_pages: Read exact page numbers from uploaded document (`query_or_arg` = e.g. '1' or '2-4')",
        "query_table_data": "- query_table_data: Compute exact sum/average/filter on CSV or table columns (`query_or_arg` = column or filter)",
        "get_weather": "- get_weather: Get real-time live weather for a specific city (`query_or_arg` = exact city name ONLY)",
        "web_search": "- web_search: Search live internet & news for recent facts/winners (`query_or_arg` = search query)",
        "scrape_website": "- scrape_website: Deeply scrape a REAL public URL (GitHub profile/repo -> drills into repos, README, file tree & code; Website -> crawls relevant sub-links; `query_or_arg` = real https:// URL ONLY, NEVER invent example.com or USERNAME)",
        "research_person": "- research_person: Multi-platform public profile search across GitHub (with repo inspection), LinkedIn public index, Indeed/career sites, and portfolios (`query_or_arg` = person name + optional handle/role)",
        "search_images": "- search_images: Search previously uploaded/described images & OCR in this session (`query_or_arg` = visual/OCR keywords)",
        "inspect_page_visual": "- inspect_page_visual: OCR/vision analysis on a PDF page (`query_or_arg` = '1: question')",
    }
    tool_catalog_lines = [_full_tool_catalog[n] for n in internal_tool_names if n in _full_tool_catalog]

    # Pure memory recall (e.g. "what was my git repo url", "what is my name") should NOT force a Step-1 web call!
    is_pure_memory_recall = bool(
        _re.search(r"\b(what was my|whats my|what's my|what is my|remember my|my previous|did i (say|give|share))\b", user_q, _re.IGNORECASE)
        and not _re.search(r"\b(scrap|scrape|crawl|fetch|check|latest|weather|search|linkedin|indeed|code|inside|analyze)\b", user_q, _re.IGNORECASE)
        and not extract_urls_from_text(user_q)
    )
    has_person_research_intent = bool(
        _re.search(r"\b(linkedin|indeed|profile|profiles|research|find (him|her|this person|about)|who is|background of)\b", user_q, _re.IGNORECASE)
    )
    explicit_external_need = bool(
        not is_pure_memory_recall
        and (
            tool_choice == "required"
            or bool(extract_urls_from_text(user_q))
            or _has_weather_intent(user_q)
            or has_person_research_intent
            or IntentRouter.RAG_PATTERNS.search(user_q)
            or IntentRouter.WEB_PATTERNS.search(user_q)
            or (orch_ctx is not None and orch_ctx.document_report is not None)
        )
    )

    observations_log: list[str] = []

    for step_idx in range(1, steps_limit + 1):
        # On Step 1, only force action='call_tool' when the query itself explicitly needs external tools (never on pure memory recall)
        if step_idx == 1 and explicit_external_need:
            action_enum = ["call_tool"]
            tool_enum = list(internal_tool_names)
        else:
            action_enum = ["call_tool", "final_answer"]
            tool_enum = ["none"] + list(internal_tool_names)

        react_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": action_enum},
                "tool_name": {"type": "string", "enum": tool_enum},
                "query_or_arg": {"type": "string", "minLength": 1},
            },
            "required": ["action", "tool_name", "query_or_arg"],
            "additionalProperties": False,
        }

        obs_section = (
            "\n\n[Completed Tool Observations So Far]:\n" + "\n\n".join(observations_log)
            if observations_log
            else "\n\n[Completed Tool Observations So Far]: None yet."
        )
        planner_msgs = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous ReAct Tool Planner. Decide the single next step as JSON.\n"
                    "Available Tools:\n" + "\n".join(tool_catalog_lines) + "\n\n"
                    "Rules:\n"
                    "1. USE SESSION MEMORY & SESSION DOCS FIRST: If [Session Conversation Memory] ALREADY answers the user's question (e.g. 'what was my URL / name?' or 'what do you remember about my app?'), set action='final_answer' directly — OR if `search_document` is available and the user asks detailed questions about a previously scraped repo/doc in this session (e.g. features, modules, use-cases of their app), call `search_document` FIRST before searching the generic web!\n"
                    "2. FOLLOW-UP SEARCH QUERY REWRITE: When calling `web_search` for a vague follow-up (like 'has anyone made such app?' or 'competitors to my app'), ALWAYS rewrite `query_or_arg` into a specific standalone descriptive search query using the actual product purpose/tech from [Session Conversation Memory] (e.g. 'offline-first QR verification warehouse audit platform alternatives') — NEVER search vague phrases like 'make such app' or a bare ambiguous name!\n"
                    "3. NEVER INVENT FAKE URLs: If the user asks to scrape/inspect their GitHub/site/repo without repeating the URL, use the EXACT real URL from [Session Conversation Memory]! NEVER output placeholder URLs like example.com or github.com/USERNAME/REPO.\n"
                    "4. DEEP GITHUB / PERSON RESEARCH: Use `scrape_website` on a real GitHub profile or repo URL to drill into repos, READMEs, file trees, and code. Use `research_person` when asked to research a person across GitHub, LinkedIn, Indeed, and portfolios.\n"
                    "5. MULTI-HOP & LOOP GUARD: If a question requires finding a fact in a document first and then calling another tool, call `search_document` on Step 1 first. Never repeat a tool call that already succeeded in [Completed Tool Observations So Far]. Once answered, set action='final_answer', tool_name='none', query_or_arg='done'."
                ),
            },
            {
                "role": "user",
                "content": f"User Request: {user_q}{memory_section}{obs_section}\n\nOutput the next ReAct JSON decision:",
            },
        ]

        planner_prompt = await service.format_messages_native(planner_msgs, tools=None)
        planner_kwargs = {
            "temperature": 0.1,
            "max_tokens": 96,
            "response_format": {"type": "json_schema", "json_schema": {"name": "react_step", "schema": react_schema}},
        }
        async with orch.vram_gate:
            plan_msg = await service.chat_with_metadata(prompt=planner_prompt, **planner_kwargs)

        plan_raw = plan_msg.content if hasattr(plan_msg, "content") else str(plan_msg)
        meta_p = getattr(plan_msg, "response_metadata", {}) or {}
        u_p = meta_p.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        total_usage["prompt_tokens"] += u_p.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += u_p.get("completion_tokens", 0)
        total_usage["total_tokens"] += u_p.get("total_tokens", 0)

        try:
            decision = json.loads(plan_raw.strip())
        except Exception:
            logger.warning(f"Planner JSON parse fallback on raw={plan_raw!r}")
            break

        action = decision.get("action")
        t_name = decision.get("tool_name")
        raw_arg = str(decision.get("query_or_arg", "")).strip()

        if action == "final_answer" or not t_name or t_name == "none" or t_name not in orch.INTERNAL_TOOL_NAMES:
            logger.info(f"✅ [Agent Step {step_idx}] Planner selected final_answer")
            break

        t_args = _map_compact_react_arg(t_name, raw_arg)
        if t_name == "scrape_website" and user_q:
            t_args["focus_query"] = user_q
        elif t_name == "research_person" and remembered_urls:
            t_args["hint_urls"] = remembered_urls
        sig_key = f"{t_name}:{json.dumps(t_args, sort_keys=True)}"
        if sig_key in seen_calls:
            logger.info(f"🔁 [Agent Step {step_idx}] Duplicate call {sig_key} prevented; moving to final synthesis")
            break
        seen_calls.add(sig_key)

        t_step0 = time.perf_counter()
        next_cit_idx = (len(orch_ctx.citations) + 1) if orch_ctx is not None else 1
        tool_obs, new_cits = await orch.execute_internal_tool_with_citations(
            tool_name=t_name,
            arguments=t_args,
            user_id=user_id,
            session_id=session_id,
            start_citation_idx=next_cit_idx,
        )
        step_ms = round((time.perf_counter() - t_step0) * 1000, 2)
        if orch_ctx is not None and new_cits:
            orch_ctx.citations.extend(new_cits)

        logger.info(f"🤖 [Agent Step {step_idx}/{steps_limit}] {t_name}({t_args}) -> {len(tool_obs)} chars in {step_ms}ms")
        agent_trace.append({
            "step": step_idx,
            "tool": t_name,
            "arguments": t_args,
            "duration_ms": step_ms,
            "citations_added": len(new_cits),
            "observation_preview": tool_obs[:240],
        })
        observations_log.append(f"<tool_output step='{step_idx}' tool='{t_name}' args='{json.dumps(t_args)}'>\n{tool_obs}\n</tool_output>")

    # Assemble clean synthesis messages with all gathered <tool_output> blocks
    synthesis_messages = [dict(m) for m in base_messages]
    if observations_log:
        obs_bundle = "<agent_tool_observations>\n" + "\n\n".join(observations_log) + "\n</agent_tool_observations>"
        for i in range(len(synthesis_messages) - 1, -1, -1):
            if synthesis_messages[i].get("role") == "user":
                synthesis_messages[i] = {
                    "role": "user",
                    "content": (
                        f"{obs_bundle}\n\n"
                        f"User Question: {synthesis_messages[i]['content']}\n"
                        "(Answer directly and completely using the verified <agent_tool_observations> above. Cite numbered sources [1], [2] inline where used.)"
                    ),
                }
                break

    if prepare_for_final_stream:
        return {
            "stream_messages": synthesis_messages,
            "content": "",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": total_usage,
            "agent_trace": agent_trace,
        }

    synth_prompt = await service.format_messages_native(synthesis_messages, tools=None)
    async with orch.vram_gate:
        ai_final = await service.chat_with_metadata(prompt=synth_prompt, **gen_kwargs)
    final_content = ai_final.content if hasattr(ai_final, "content") else str(ai_final)
    meta_f = getattr(ai_final, "response_metadata", {}) or {}
    u_f = meta_f.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    total_usage["prompt_tokens"] += u_f.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += u_f.get("completion_tokens", 0)
    total_usage["total_tokens"] += u_f.get("total_tokens", 0)

    return {
        "stream_messages": None,
        "content": final_content,
        "tool_calls": None,
        "finish_reason": meta_f.get("finish_reason", "stop"),
        "usage": total_usage,
        "agent_trace": agent_trace,
    }


@llmRouter.post("/chat/completions")
async def completion(
    req: Request,
    request: ChatRequest | None = Form(default=None, description="ChatRequest JSON"),
    image: UploadFile | None = File(
        default=None,
        description="Optional file or image upload (auto-detects Images, PDF, Scanned PDF, DOCX, XLSX, CSV, ZIP, JSON, TXT, Code strictly in RAM)",
    ),
    agentic_mode: bool | None = Form(
        default=None,
        description="Optional Form override: True enables autonomous multi-step ReAct Agentic tool loop",
    ),
    max_agent_steps: int | None = Form(
        default=None,
        description="Optional Form override: Max ReAct agent hops (1..6) when agentic_mode=True",
    ),
    service: LlmService = Depends(get_repo),
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
        if agentic_mode is not None:
            request.agentic_mode = agentic_mode
        if max_agent_steps is not None:
            request.max_agent_steps = max(1, min(int(max_agent_steps), 6))
        if image is not None and image.filename:
            uploaded_bytes = await image.read()
            uploaded_filename = image.filename
            if not uploaded_bytes:
                raise InvalidFileError("Uploaded file is empty")
            if len(uploaded_bytes) > MAX_IMAGE_UPLOAD_BYTES:
                raise FileTooLargeError("Uploaded file exceeds 15MB limit")
            uploaded_kind, _ = detect_upload_kind(uploaded_bytes, uploaded_filename)

    active_model = await _require_loaded_model(service, request.model)
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

    # Check if request includes image and ensure vision model is loaded (or remote url available)
    has_image_input = (uploaded_kind == "image") or bool(getattr(request, "image_base64", None))
    if not has_image_input and request.messages:
        for m in request.messages:
            if isinstance(m.content, list):
                for p in m.content:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        has_image_input = True
                        break
    if has_image_input:
        _require_loaded_vision_model(_orchestrator)

    # ============================================
    # 1. ORCHESTRATION (RAG + WEB SEARCH + DIRECT URL SCRAPE + VISION + UNIVERSAL DOC PARSER)
    # ============================================
    is_agentic = bool(getattr(request, "agentic_mode", False))
    max_agent_steps = int(getattr(request, "max_agent_steps", 3) or 3)
    orch_ctx = None
    should_orchestrate = settings.ORCH_ENABLED and (
        is_agentic
        or getattr(request, "use_orchestrator", True)
        or getattr(request, "use_web", None) is True
        or getattr(request, "use_rag", None) is True
        or uploaded_bytes is not None
        or getattr(request, "image_base64", None) is not None
    )
    effective_force_web = getattr(request, "use_web", None)
    if effective_force_web is None and request.tools and request.tool_choice != "none" and not is_agentic:
        # Do not pre-empt explicit external function-calling tools with background web scraping unless use_web=True
        effective_force_web = False

    if should_orchestrate:
        orch_ctx = await _orchestrator.prepare_context(
            request.messages,
            file_bytes=uploaded_bytes,
            filename=uploaded_filename,
            image_base64=getattr(request, "image_base64", None),
            force_web=effective_force_web,
            force_rag=getattr(request, "use_rag", None),
            user_id=tenant_user_id,
            session_id=tenant_session_id,
            model_name=active_model,
            agentic_mode=is_agentic,
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

    if is_agentic:
        has_tenant_docs_for_tools = (
            (getattr(request, "use_rag", None) is not False)
            and (
                (orch_ctx is not None and orch_ctx.document_report is not None)
                or await _orchestrator.memory.has_tenant_documents(tenant_user_id, tenant_session_id)
            )
        )
        has_tenant_imgs_for_tools = (
            has_image_input
            or await _orchestrator.memory.has_tenant_images(tenant_user_id, tenant_session_id)
        )
        q_for_tools = orch_ctx.user_query if orch_ctx is not None else ""
        active_tools = _merge_agentic_and_client_tools(
            _orchestrator,
            request.tools,
            allow_web=(getattr(request, "use_web", None) is not False),
            allow_rag=has_tenant_docs_for_tools,
            has_local_images=has_tenant_imgs_for_tools,
            allow_weather=_has_weather_intent(q_for_tools),
        )
    else:
        active_tools = request.tools
    base_messages_for_agent = [dict(m) for m in raw_messages]
    if not is_agentic:
        raw_messages = _inject_tools_into_messages(raw_messages, active_tools, request.tool_choice)
        prompt = await service.format_messages_native(raw_messages, tools=active_tools)
    else:
        prompt = ""

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
        if is_agentic and active_tools:
            loop_pre = await _execute_hybrid_react_loop(
                service=service,
                orch=_orchestrator,
                orch_ctx=orch_ctx,
                base_messages=base_messages_for_agent,
                active_tools=active_tools,
                tool_choice=request.tool_choice,
                gen_kwargs=gen_kwargs,
                user_id=tenant_user_id,
                session_id=tenant_session_id,
                max_steps=max_agent_steps,
                is_agentic=is_agentic,
                client_tools=request.tools,
                prepare_for_final_stream=True,
            )
            if loop_pre["stream_messages"] is not None:
                prompt = await service.format_messages_native(loop_pre["stream_messages"])
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
                    async for token in service.stream_llm(prompt, chat_id=stream_task_id, **gen_kwargs):
                        if await req.is_disconnected():
                            logger.info(f"Client disconnected mid-stream; cancelling {stream_task_id}")
                            await service.cancel_request(stream_task_id)
                            return
                        collected.append(token)
                        yield chunk({"content": token})
                completed_normally = True
                yield chunk({}, finish_reason="stop")
                if include_usage:
                    yield chunk({}, finish_reason=None, usage=service.pop_last_usage(stream_task_id))
                else:
                    service.pop_last_usage(stream_task_id)
                yield "data: [DONE]\n\n"
            finally:
                if not completed_normally:
                    await service.cancel_request(stream_task_id)
                    service.pop_last_usage(stream_task_id)

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

    agent_trace: list[dict[str, Any]] | None = None
    if is_agentic or active_tools:
        t_ag0 = time.perf_counter()
        loop_res = await _execute_hybrid_react_loop(
            service=service,
            orch=_orchestrator,
            orch_ctx=orch_ctx,
            base_messages=base_messages_for_agent,
            active_tools=active_tools,
            tool_choice=request.tool_choice,
            gen_kwargs=gen_kwargs,
            user_id=tenant_user_id,
            session_id=tenant_session_id,
            max_steps=max_agent_steps if is_agentic else 1,
            is_agentic=is_agentic,
            client_tools=request.tools,
        )
        content = loop_res["content"]
        tool_calls = loop_res["tool_calls"]
        finish_reason = loop_res["finish_reason"]
        exact_usage = loop_res["usage"]
        agent_trace = loop_res["agent_trace"] if is_agentic else (loop_res["agent_trace"] or None)
        if orch_ctx is not None and is_agentic:
            orch_ctx.metrics_ms["agent_loop_ms"] = round((time.perf_counter() - t_ag0) * 1000, 2)
    else:
        async with _orchestrator.vram_gate:
            ai_msg = await service.chat_with_metadata(
                prompt=prompt,
                **gen_kwargs
            )
        content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
        meta = getattr(ai_msg, "response_metadata", {}) or {}
        exact_usage = meta.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        tool_calls = None
        finish_reason = meta.get("finish_reason", "stop")

    if content and not tool_calls:
        content = strip_ungrounded_citations(content, orch_ctx.citations if orch_ctx else None)

    if orch_ctx is not None and content:
        await _orchestrator.memory.record_chat_turn(
            query=orch_ctx.user_query,
            answer=content,
            model_used=request.model,
            route=orch_ctx.route,
            user_id=tenant_user_id,
            session_id=tenant_session_id,
        )

    logger.info(f"✅ LLM response received (agentic={is_agentic}, steps={len(agent_trace or [])}): {(content or '')[:100]}...")

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
        agentic_mode=is_agentic,
        agent_trace=agent_trace,
        citations=[c.model_dump() for c in orch_ctx.citations] if (orch_ctx and orch_ctx.citations) else None,
    )


# @llmRouter.post("/chat/completions")
# async def completion(request: ChatRequest, req: Request, service: LlmService = Depends(get_repo)):
#     try:
#         logger.info(f"Chat request - Model: {request.model}, Messages: {len(request.messages)}")
        
#         # ============================================
#         # 1. CHECK IF REPOSITORY IS READY
#         # ============================================
#         if not await service.is_ready():
#             logger.warning("Repository not ready - attempting to initialize...")
#             try:
#                 await service.initialize(model_name=request.model)
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
        
#         response = await service.chat(
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
async def load_model(model_name: str, service: LlmService = Depends(get_repo)):
    """
    Load a specific model into the repository.
    If the model is already loaded, just return a message saying so.
    """
    is_ready = await service.is_ready()
    loaded_models = await service.get_loaded_models_list()

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
        await service.shutdown()
        logger.info(f"Model '{current_model}' unloaded successfully")

    logger.info(f"Loading model '{model_name}'...")
    await service.initialize(model_name=model_name)

    return {
        "status": "success",
        "message": f"Model '{model_name}' loaded successfully",
        "model": model_name,
        "ready": await service.is_ready(),
        "loaded_models": await service.get_loaded_models_list(),
    }


# ============================================
# UNLOAD MODEL ENDPOINT
# ============================================

@llmRouter.post("/unload_model")
async def unload_model(service: LlmService = Depends(get_repo)):
    """
    Unload the currently loaded model.
    If no model is loaded, just return a message saying so.
    """
    is_ready = await service.is_ready()
    loaded_models = await service.get_loaded_models_list()

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
    await service.shutdown()
    logger.info(f"Model '{current_model}' unloaded successfully")

    return {
        "status": "success",
        "message": f"✅ Model '{current_model}' unloaded successfully",
        "unloaded_model": current_model,
        "ready": await service.is_ready(),
        "loaded_models": await service.get_loaded_models_list(),
    }


# ============================================
# HEALTH CHECK ENDPOINT
# ============================================

@llmRouter.get("/health")
async def health_check(service: LlmService = Depends(get_repo)):
    """Check if the LLM service is ready."""
    is_ready = await service.is_ready()
    loaded_models = await service.get_loaded_models_list()
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
async def get_dir_models(service: LlmService = Depends(get_repo)):
    """Get list of available models from the directory."""
    models = await service.get_models_cdir()
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
async def get_active_model(service: LlmService = Depends(get_repo)):
    """View the currently active/loaded model and runtime hardware configuration."""
    is_ready = await service.is_ready()
    loaded_models = await service.get_loaded_models_list()
    return {
        "ready": is_ready,
        "active_model": loaded_models[0] if loaded_models else None,
        "loaded_models": loaded_models,
        "orchestrator_enabled": settings.ORCH_ENABLED,
        "vision_model": settings.ORCH_VISION_MODEL,
        "config": {
            "n_ctx": service.n_ctx,
            "n_gpu_layers": service.n_gpu_layers,
            "n_batch": service.n_batch,
            "n_threads": service.n_threads,
            "continuous_batch_slots": service.continuous_batch_slots,
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
# VISION MODEL LIFECYCLE ENDPOINTS
# ============================================

@llmRouter.post("/vision/load_model")
async def load_vision_model_endpoint(
    model_name: str | None = Form(default=None, description="Optional vision GGUF model filename"),
    mmproj_name: str | None = Form(default=None, description="Optional vision mmproj projector filename"),
    gpu_layers: int | None = Form(default=None, description="GPU layers (0=CPU resident, -1=GPU)"),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Load a Vision GGUF model into memory and keep it resident for rapid subsequent OCR/vision inference."""
    result = await orch.vision_tool.load_model(
        model_name=model_name,
        mmproj_name=mmproj_name,
        gpu_layers=gpu_layers,
    )
    return {"status": "ok", "message": "Vision model loaded successfully", "data": result}


@llmRouter.post("/vision/unload_model")
async def unload_vision_model_endpoint(
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Unload resident Vision model and free system memory."""
    result = await orch.vision_tool.unload_model()
    return {"status": "ok", "message": result.get("message", "Vision model unloaded"), "data": result}


@llmRouter.get("/vision/models")
async def list_vision_models_endpoint(
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """List available vision GGUF models and mmproj projectors strictly from VISION_MODEL_PATH."""
    return orch.vision_tool.list_vision_models()


@llmRouter.get("/vision/active_model")
async def get_active_vision_model_endpoint(
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Get the currently loaded resident vision model and status."""
    return orch.vision_tool.get_active_model()


# ============================================
# IN-RAM IMAGE UPLOAD ENDPOINTS (NO DISK SAVE)
# ============================================

MAX_IMAGE_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB RAM guard


@llmRouter.post("/vision/describe")
async def describe_uploaded_image(
    req: Request,
    image: UploadFile = File(..., description="Image file processed strictly in RAM"),
    prompt: str = Form(default="", description="Optional focus hint for the Vision model"),
    user_id: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """
    Pure Vision (VLM) OCR & description endpoint:
    Upload an image -> decode in RAM with Vision model (local or VISION_API_URL) -> index in ChromaDB isolated by (user_id, session_id).
    """
    _require_loaded_vision_model(orch)
    uid, sid = _resolve_tenant(req, explicit_user_id=user_id, explicit_session_id=session_id)
    image_bytes = await image.read()
    if not image_bytes:
        raise InvalidImageError("Uploaded image is empty")
    if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
        raise ImageTooLargeError("Image exceeds 15MB limit")

    t_vis0 = time.perf_counter()
    report = await orch.vision_tool.describe_and_index(
        image_bytes=image_bytes,
        user_hint=prompt,
        user_id=uid,
        session_id=sid,
    )
    vision_ms = round((time.perf_counter() - t_vis0) * 1000, 2)

    return {
        "status": "ok",
        "user_id": uid,
        "session_id": sid,
        "vision_report": report.model_dump(),
        "metrics_ms": {"vision_ms": vision_ms},
    }


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
    agentic_mode: bool = Form(
        default=False,
        description="False = Fast 1-pass pre-fetch mode; True = Multi-step ReAct Agentic mode where LLM chooses & chains DB/Web/Weather tools",
    ),
    max_agent_steps: int = Form(default=3, description="Max autonomous tool-calling iterations when agentic_mode=True (1..6)"),
    use_web: bool | None = Form(default=None),
    use_rag: bool | None = Form(default=None),
    temperature: float | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    service: LlmService = Depends(get_repo),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """Chat with optional uploaded file/image + chained JSON list of system prompts + main user prompt (strictly isolated by user_id & session_id)."""
    active_model = await _require_loaded_model(service, model)
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
        up_kind, _ = detect_upload_kind(uploaded_bytes, uploaded_filename)
        if up_kind == "image":
            _require_loaded_vision_model(orch)

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
        agentic_mode=agentic_mode,
    )
    gen_kwargs = {k: v for k, v in {"temperature": temperature, "max_tokens": max_tokens}.items() if v is not None}

    agent_trace: list[dict[str, Any]] | None = None
    if agentic_mode:
        has_up_docs = (use_rag is not False) and (
            orch_ctx.document_report is not None
            or await orch.memory.has_tenant_documents(uid, sid)
        )
        has_up_imgs = (
            orch_ctx.vision_report is not None
            or await orch.memory.has_tenant_images(uid, sid)
        )
        active_upload_tools = _merge_agentic_and_client_tools(
            orch,
            None,
            allow_web=(use_web is not False),
            allow_rag=has_up_docs,
            has_local_images=has_up_imgs,
            allow_weather=_has_weather_intent(prompt),
        )
        t_ag0 = time.perf_counter()
        loop_res = await _execute_hybrid_react_loop(
            service=service,
            orch=orch,
            orch_ctx=orch_ctx,
            base_messages=orch_ctx.enriched_messages,
            active_tools=active_upload_tools,
            tool_choice=None,
            gen_kwargs=gen_kwargs,
            user_id=uid,
            session_id=sid,
            max_steps=max_agent_steps,
            is_agentic=True,
            client_tools=None,
        )
        content = loop_res["content"]
        extracted_calls = loop_res["tool_calls"]
        finish_reason = loop_res["finish_reason"]
        agent_trace = loop_res["agent_trace"]
        orch_ctx.metrics_ms["agent_loop_ms"] = round((time.perf_counter() - t_ag0) * 1000, 2)
    else:
        formatted_prompt = await service.format_messages_native(orch_ctx.enriched_messages)
        async with orch.vram_gate:
            response = await service.chat(prompt=formatted_prompt, **gen_kwargs)
        content = response if isinstance(response, str) else str(response)
        extracted_calls = None
        finish_reason = "stop"

    if content and not extracted_calls:
        content = strip_ungrounded_citations(content, orch_ctx.citations)

    if content:
        await orch.memory.record_chat_turn(
            query=orch_ctx.user_query,
            answer=content,
            model_used=active_model,
            route=orch_ctx.route,
            user_id=uid,
            session_id=sid,
        )
    msg_out: dict[str, Any] = {"role": "assistant", "content": None if extracted_calls else content}
    if extracted_calls:
        msg_out["tool_calls"] = extracted_calls
    return {
        "id": f"chatcmpl-{uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": active_model,
        "user_id": uid,
        "session_id": sid,
        "system_prompts_count": len(sys_chain),
        "agentic_mode": agentic_mode,
        "agent_trace": agent_trace,
        "choices": [{"index": 0, "message": msg_out, "finish_reason": finish_reason}],
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
    agentic_mode: bool = Form(
        default=False,
        description="False = Fast AST analysis; True = Multi-step ReAct Agentic mode with AST/RAG/Web tools",
    ),
    max_agent_steps: int = Form(default=3, description="Max ReAct agent hops when agentic_mode=True (1..6)"),
    use_web: bool | None = Form(default=False),
    use_rag: bool | None = Form(default=True),
    temperature: float | None = Form(default=0.2),
    max_tokens: int | None = Form(default=1024),
    service: LlmService = Depends(get_repo),
    orch: ChatOrchestrator = Depends(get_orchestrator),
):
    """
    Dedicated Coding & AST Analysis Endpoint:
    - Accepts code files in ANY language in the world (Python, JS/TS, C/C++, Rust, Go, Java, C#, PHP, Ruby, Solidity, SQL, Bash, etc.).
    - Splits code into AST nodes in RAM and indexes every function/class/symbol into ChromaDB (`docs_rag`) isolated by `(user_id, session_id)`.
    - Automatically compresses large code files into an AST Skeleton + top relevant AST symbols when code length exceeds the context window budget.
    """
    from App.core.orchestrator import UniversalCodeASTProcessor, ParsedPage, ParsedDocumentReport

    active_model = await _require_loaded_model(service, model)
    uid, sid = _resolve_tenant(req, explicit_user_id=user_id, explicit_session_id=session_id)

    t0 = time.perf_counter()
    ast_report: dict | None = None
    code_context_block = ""
    vision_report_obj = None
    is_image_code = False

    # Calculate safe token-saving character budget based on active model's per-sequence KV context window
    per_seq_ctx = max(512, service.n_ctx // max(1, service.continuous_batch_slots))
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
            _require_loaded_vision_model(orch)
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
        agentic_mode=agentic_mode,
    )
    gen_kwargs = {k: v for k, v in {"temperature": temperature, "max_tokens": max_tokens}.items() if v is not None}
    code_agent_trace: list[dict[str, Any]] | None = None
    code_tool_calls: list[dict] | None = None
    if agentic_mode:
        has_code_docs = (use_rag is not False) and (
            code_file is not None
            or await orch.memory.has_tenant_documents(uid, sid)
        )
        has_code_imgs = is_image_code or await orch.memory.has_tenant_images(uid, sid)
        active_code_tools = _merge_agentic_and_client_tools(
            orch,
            None,
            allow_web=(use_web is not False or agentic_mode),
            allow_rag=has_code_docs,
            has_local_images=has_code_imgs,
            allow_weather=_has_weather_intent(prompt),
        )
        t_ag0 = time.perf_counter()
        loop_res = await _execute_hybrid_react_loop(
            service=service,
            orch=orch,
            orch_ctx=orch_ctx,
            base_messages=orch_ctx.enriched_messages,
            active_tools=active_code_tools,
            tool_choice=None,
            gen_kwargs=gen_kwargs,
            user_id=uid,
            session_id=sid,
            max_steps=max_agent_steps,
            is_agentic=True,
            client_tools=None,
        )
        content = loop_res["content"]
        code_tool_calls = loop_res["tool_calls"]
        meta = {"finish_reason": loop_res["finish_reason"]}
        exact_usage = loop_res["usage"]
        code_agent_trace = loop_res["agent_trace"]
        orch_ctx.metrics_ms["agent_loop_ms"] = round((time.perf_counter() - t_ag0) * 1000, 2)
    else:
        formatted_prompt = await service.format_messages_native(orch_ctx.enriched_messages)
        async with orch.vram_gate:
            ai_msg = await service.chat_with_metadata(prompt=formatted_prompt, **gen_kwargs)
        content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
        meta = getattr(ai_msg, "response_metadata", {}) or {}
        exact_usage = meta.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    if content and not code_tool_calls:
        content = strip_ungrounded_citations(content, orch_ctx.citations)

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
        "agentic_mode": agentic_mode,
        "agent_trace": code_agent_trace,
        "choices": [
            {
                "index": 0,
                "message": (
                    {"role": "assistant", "content": None, "tool_calls": code_tool_calls}
                    if code_tool_calls
                    else {"role": "assistant", "content": content}
                ),
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