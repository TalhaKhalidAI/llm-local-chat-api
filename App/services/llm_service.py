# App/services/llm_service.py
"""
LLM Service Layer
=================
Owns all business logic, model lifecycle coordination, readiness validation,
and chat/streaming orchestration.
Delegates direct engine operations to llmRepository (Data/Engine Access Layer).
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from App.core.exceptions import ModelNotLoadedError
from App.repository.llmRepository import llmChatRepo, llm_repo

logger = logging.getLogger(__name__)


class LlmService:
    """
    Business logic layer for LLM operations.
    Encapsulates state checks, model loading/unloading rules, and prompt execution.
    """

    def __init__(self, repo: Optional[llmChatRepo] = None):
        self.repo = repo or llm_repo

    @property
    def is_initialized(self) -> bool:
        return self.repo._is_initialized

    @property
    def n_ctx(self) -> int:
        return self.repo.n_ctx

    @property
    def n_gpu_layers(self) -> int:
        return self.repo.n_gpu_layers

    @property
    def n_batch(self) -> int:
        return self.repo.n_batch

    @property
    def n_threads(self) -> int:
        return self.repo.n_threads

    @property
    def continuous_batch_slots(self) -> int:
        return self.repo.continuous_batch_slots

    async def is_ready(self) -> bool:
        """Check if LLM engine is initialized and worker threads are healthy."""
        return self.repo._is_initialized and await self.repo.is_healthy()

    async def get_loaded_models_list(self) -> List[str]:
        """Get list of currently resident model names."""
        return await self.repo.get_loaded_models()

    async def get_models_cdir(self) -> List[Dict[str, str]]:
        """Get available models in the configured directory."""
        models = await self.repo._get_model_in_current_dirs()
        logger.info(f"Found {len(models)} model(s) in directory")
        return models

    async def initialize(self, model_name: Optional[str] = None, **load_kwargs):
        """
        Business rule: Validate and load the requested model.
        Initializes background processing workers.
        """
        if self.repo._is_initialized:
            logger.info("LLM Service/Repository already initialized")
            return self

        model_to_load = model_name or self.repo._model_name
        if not model_to_load:
            raise ModelNotLoadedError("model_name is required for initialization")

        await self.repo._load_model(model_name=model_to_load, **load_kwargs)
        await self.repo.start()

        self.repo._is_initialized = True
        logger.info(f"LLM Service initialized with model: {model_to_load}")
        return self

    async def shutdown(self):
        """Business rule: Clean shutdown and unload all resident models."""
        if self.repo._is_initialized:
            await self.repo.stop()
            await self.repo.unload_all_models()
            self.repo._is_initialized = False
            logger.info("LLM Service/Repository shut down successfully")

    async def format_messages_native(self, messages: list, tools: Optional[list] = None) -> str:
        """Apply model chat template."""
        return await self.repo.format_messages_native(messages, tools=tools)

    async def chat_with_metadata(self, prompt: str, **gen_kwargs):
        """Send a chat request and return the full AIMessage with token usage."""
        if not await self.is_ready():
            raise ModelNotLoadedError("No model is currently loaded. Call POST /app/v1/load_model first.")
        return await self.repo.chat_llm(prompt, **gen_kwargs)

    async def chat(self, prompt: str, **gen_kwargs) -> str:
        """Send chat request and return message content string."""
        if not await self.is_ready():
            raise ModelNotLoadedError("No model is currently loaded. Call POST /app/v1/load_model first.")
        response = await self.repo.chat_llm(prompt, **gen_kwargs)
        return response.content if hasattr(response, 'content') else str(response)

    def stream_llm(self, prompt: str, chat_id: Optional[str] = None, **gen_kwargs) -> AsyncGenerator[str, None]:
        """Stream tokens directly from the engine."""
        return self.repo.stream_llm(prompt, chat_id=chat_id, **gen_kwargs)

    async def cancel_request(self, chat_id: str):
        """Cancel in-flight generation task."""
        return await self.repo.cancel_request(chat_id)

    def pop_last_usage(self, chat_id: str):
        """Retrieve usage stats for streaming task."""
        return self.repo.pop_last_usage(chat_id)


# Application-wide service singleton
llm_service = LlmService(llm_repo)

def get_llm_service() -> LlmService:
    """FastAPI dependency for LlmService."""
    return llm_service
