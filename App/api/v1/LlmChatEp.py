from App.repository.llmRepository import llmChatRepo
from fastapi import APIRouter, HTTPException, status, Request, Depends
from App.models.llmModels import ChatChoice, ChatMessage, ChatRequest, ChatResponse
from App.core.settings import settings
from fastapi.responses import StreamingResponse
from uuid import uuid4
import time
import json
import traceback
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

def split_large_messages(messages, splitter, max_chunk_size: int = 4000) -> str:
    """
    Build prompt from messages with automatic splitting of large content.
    """
    prompt = ""
    
    for msg in messages:
        content = msg.content
        
        # Check if content needs splitting
        if len(content) > max_chunk_size:
            logger.info(f"📝 Splitting {msg.role} message: {len(content)} chars -> chunks")
            chunks = splitter.split_text(content)
            
            # Add each chunk with part numbering
            for i, chunk in enumerate(chunks):
                part_label = f" (Part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
                
                if msg.role == "system":
                    prompt += f"[SYSTEM]{part_label} {chunk}\n"
                elif msg.role == "user":
                    prompt += f"[USER]{part_label} {chunk}\n"
                elif msg.role == "assistant":
                    prompt += f"[ASSISTANT]{part_label} {chunk}\n"
            
            logger.info(f"✅ Split into {len(chunks)} chunks")
        else:
            # Normal content - no splitting needed
            if msg.role == "system":
                prompt += f"[SYSTEM] {content}\n"
            elif msg.role == "user":
                prompt += f"[USER] {content}\n"
            elif msg.role == "assistant":
                prompt += f"[ASSISTANT] {content}\n"
    
    prompt += "[ASSISTANT] "
    return prompt

@llmRouter.post("/chat/completions")
async def completion(request: ChatRequest, req: Request, repo: llmChatRepo = Depends(get_repo)):
    try:
        logger.info(f"Chat request - Model: {request.model}, Messages: {len(request.messages)}")
        
        # ============================================
        # 1. CHECK IF REPOSITORY IS READY
        # ============================================
        if not await repo.is_ready():
            logger.warning("Repository not ready - attempting to initialize...")
            try:
                await repo.initialize(model_name=request.model)
                logger.info(f"Repository initialized with model: {request.model}")
            except Exception as e:
                logger.error(f"Failed to initialize repository: {e}")
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
                                content=f"Error: Model '{request.model}' not loaded. Error: {str(e)}"
                            ),
                            finish_reason="error"
                        )
                    ],
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )
        
        # ============================================
        # 2. BUILD PROMPT FROM MESSAGES WITH SPLITTING
        # ============================================
        prompt = split_large_messages(request.messages, text_splitter)
        
        if not prompt or prompt == "[ASSISTANT] ":
            prompt = "[USER] Hello\n[ASSISTANT] "
        
        # Log prompt size
        prompt_chars = len(prompt)
        prompt_tokens_estimate = prompt_chars // 4
        logger.info(f"📝 Prompt: {prompt_chars} chars (~{prompt_tokens_estimate} tokens)")
        logger.info(f"📄 Preview: {prompt[:150]}...")

        # ============================================
        # 3. CALL THE LLM with safe parameter handling
        # ============================================
        # Build kwargs safely from request parameters
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

        if settings.STREAM_RESPONSE:
            response_id = f"chatcmpl-{uuid4().hex[:8]}"
            created = int(time.time())

            async def stream_response():
                def chunk(delta, finish_reason=None):
                    return (
                        "data: "
                        + json.dumps({
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }],
                        })
                        + "\n\n"
                    )

                yield chunk({"role": "assistant"})
                async for token in repo.stream_llm(prompt, **gen_kwargs):
                    yield chunk({"content": token})
                yield chunk({}, finish_reason="stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        
        response = await repo.chat(
            prompt=prompt,
            **gen_kwargs
        )
        
        content = response if isinstance(response, str) else str(response)
        
        logger.info(f"✅ LLM response received: {content[:100]}...")
        
        # Calculate usage
        completion_tokens = len(content.split())
        
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
                        content=content
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens_estimate,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens_estimate + completion_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred during chat with LLM: {str(e)}"
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
    try:
        # Check if repository is ready and what models are loaded
        is_ready = await repo.is_ready()
        loaded_models = await repo.get_loaded_models_list()
        
        # If model is already loaded, return early
        if is_ready and model_name in loaded_models:
            logger.info(f"Model '{model_name}' is already loaded")
            return {
                "status": "already_loaded",
                "message": f"Model '{model_name}' is already loaded and ready to use",
                "model": model_name,
                "ready": True,
                "loaded_models": loaded_models
            }
        
        # If a different model is loaded, unload it first
        if is_ready and loaded_models:
            current_model = loaded_models[0]
            logger.info(f"Unloading current model '{current_model}' to load '{model_name}'")
            await repo.shutdown()
            logger.info(f"Model '{current_model}' unloaded successfully")
        
        # Load the new model
        logger.info(f"Loading model '{model_name}'...")
        await repo.initialize(model_name=model_name)
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' loaded successfully",
            "model": model_name,
            "ready": await repo.is_ready(),
            "loaded_models": await repo.get_loaded_models_list()
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {e}"
        )

# ============================================
# UNLOAD MODEL ENDPOINT
# ============================================

@llmRouter.post("/unload_model")
async def unload_model(repo: llmChatRepo = Depends(get_repo)):
    """
    Unload the currently loaded model.
    If no model is loaded, just return a message saying so.
    """
    try:
        # Check if any model is loaded
        is_ready = await repo.is_ready()
        loaded_models = await repo.get_loaded_models_list()
        
        # If no model is loaded, return early
        if not is_ready or not loaded_models:
            logger.info("No model is currently loaded")
            return {
                "status": "already_unloaded",
                "message": "ℹ️ No model is currently loaded. Nothing to unload.",
                "ready": False,
                "loaded_models": []
            }
        
        # Get the current model name before unloading
        current_model = loaded_models[0]
        
        # Unload the model
        logger.info(f"Unloading model '{current_model}'...")
        await repo.shutdown()
        logger.info(f"Model '{current_model}' unloaded successfully")
        
        return {
            "status": "success",
            "message": f"✅ Model '{current_model}' unloaded successfully",
            "unloaded_model": current_model,
            "ready": await repo.is_ready(),
            "loaded_models": await repo.get_loaded_models_list()
        }
        
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unloading model: {e}"
        )

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================

@llmRouter.get("/health")
async def health_check(repo: llmChatRepo = Depends(get_repo)):
    """
    Check if the LLM service is ready.
    """
    try:
        is_ready = await repo.is_ready()
        loaded_models = await repo.get_loaded_models_list()
        
        return {
            "status": "healthy" if is_ready else "unhealthy",
            "ready": is_ready,
            "models_loaded": len(loaded_models),
            "loaded_models": loaded_models,
            "message": "LLM service is ready" if is_ready else "No model loaded. Call /load_model first."
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "ready": False,
            "error": str(e)
        }


# ============================================
# MODELS LIST ENDPOINT
# ============================================

@llmRouter.get("/models")
async def get_dir_models(repo: llmChatRepo = Depends(get_repo)):
    """
    Get list of available models from the directory.
    """
    try:
        models = await repo.get_models_cdir()
        return {
            "object": "list",
            "data": [
                {
                    "id": model["filename"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "path": model["path"]
                }
                for model in models
            ]
        }
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading models: {e}"
        )