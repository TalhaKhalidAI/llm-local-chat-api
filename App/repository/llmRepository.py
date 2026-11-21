from App.core.AsyncLLM import AsyncLLM
from App.core.exceptions import ModelNotLoadedError
import logging
from typing import List, Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)

class llmChatRepo(AsyncLLM):
    """
    Repository class for LLM Chat operations.
    Extends AsyncLLM with initialization and model management.
    Exceptions propagate directly to main.py exception handlers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_initialized = False
        self._model_name = kwargs.get("model_name", None)
        
    async def initialize(self, model_name: Optional[str] = None, **load_kwargs):
        if self._is_initialized:
            logger.info("Repository already initialized")
            return self
        
        model_to_load = model_name or self._model_name
        if not model_to_load:
            raise ModelNotLoadedError("model_name is required for initialization")
        
        await self._load_model(model_name=model_to_load, **load_kwargs)
        await self.start()
        
        self._is_initialized = True
        logger.info(f"Repository initialized with model: {model_to_load}")
        return self

    async def get_models_cdir(self) -> List[Dict[str, str]]:
        models = await self._get_model_in_current_dirs()
        logger.info(f"Found {len(models)} model(s) in directory")
        return models
    
    async def get_loaded_models_list(self) -> List[str]:
        return await self.get_loaded_models()
    
    async def is_ready(self) -> bool:
        return self._is_initialized and await self.is_healthy()
    
    async def chat_with_metadata(self, prompt: str, **gen_kwargs):
        """Send a chat request and return the full AIMessage (including exact token usage)."""
        if not await self.is_ready():
            raise ModelNotLoadedError("No model is currently loaded. Call POST /app/v1/load_model first.")
        return await self.chat_llm(prompt, **gen_kwargs)

    async def chat(self, prompt: str, **gen_kwargs) -> str:
        if not await self.is_ready():
            raise ModelNotLoadedError("No model is currently loaded. Call POST /app/v1/load_model first.")
        
        response = await self.chat_llm(prompt, **gen_kwargs)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def shutdown(self):
        """Clean shutdown of the repository."""
        if self._is_initialized:
            await self.stop()
            await self.unload_all_models()
            self._is_initialized = False
            logger.info("Repository shut down successfully")