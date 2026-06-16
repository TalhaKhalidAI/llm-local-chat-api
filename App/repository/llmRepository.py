from App.core.AsyncLLM import AsyncLLM
import logging
from typing import List, Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)

class llmChatRepo(AsyncLLM):
    """
    Repository class for LLM Chat operations.
    Extends AsyncLLM with initialization and model management.
    """

    def __init__(self, **kwargs):
        """
        Initialize AsyncLLM with configurable parameters.
        
        Kwargs:
            model_path: Path to directory containing .gguf model files.
            temperature: Sampling temperature (0.0-2.0). Lower = more deterministic.
            top_p: Nucleus sampling probability (0.0-1.0).
            top_k: Top-k sampling (0 = disabled).
            n_predict: Maximum tokens to generate.
            n_gpu_layers: GPU offload layers (-1=auto, 0=CPU only).
            n_ctx: Context window size in tokens.
            n_threads: Number of CPU threads for inference.
            n_batch: Batch size for prompt processing.
            streaming: Enable streaming by default.
            repeat_penalty: Penalty for repeating tokens.
            verbose: Enable llama.cpp verbose logging.
            stop: List of stop tokens.
            queue_maxsize: Maximum queue capacity.
            default_timeout: Request timeout in seconds.
            auto_restart_worker: Auto-restart worker on crash.
            worker_exception_callback: Function called on worker crash.
            reject_on_full_queue: Raise QueueFull instead of waiting.
            graceful_shutdown: Wait for pending requests on shutdown.
            enable_metrics: Collect request metrics.
            enable_cancellation: Allow cancelling in-flight requests.
            available_ram: Minimum RAM required (MB) before loading model.
        """
        super().__init__(**kwargs)
        self._is_initialized = False
        self._model_name = kwargs.get("model_name", None)
        
    async def initialize(self, model_name: Optional[str] = None, **load_kwargs):
        """
        Initialize the LLM by loading the model and starting the worker.
        
        Args:
            model_name: Name of the model to load (if not provided, uses self._model_name)
            **load_kwargs: Additional loading parameters (temperature, n_gpu_layers, etc.)
        
        Returns:
            self: The initialized repository instance
        """
        try:
            if self._is_initialized:
                logger.info("Repository already initialized")
                return self
            
            # Use provided model_name or fallback to stored
            model_to_load = model_name or self._model_name
            
            if not model_to_load:
                raise ValueError("model_name is required for initialization")
            
            # Load the model
            await self._load_model(model_name=model_to_load, **load_kwargs)
            
            # Start the worker
            await self.start()
            
            self._is_initialized = True
            logger.info(f"Repository initialized with model: {model_to_load}")
            return self
            
        except Exception as e:
            logger.error(f"Failed to initialize repository: {e}")
            raise ValueError(f"Error initializing model due to: {e}")

    async def get_models_cdir(self) -> List[Dict[str, str]]:
        """
        Get all models from the current directory.
        
        Returns:
            List of model information dictionaries with keys: path, filename, directory
        """
        try:
            models = await self._get_model_in_current_dirs()
            logger.info(f"Found {len(models)} model(s) in directory")
            return models
        except Exception as e:
            logger.error(f"Error getting models from directory: {e}")
            raise ValueError(f"Error loading models due to: {e}")
    
    async def get_loaded_models_list(self) -> List[str]:
        """
        Get list of currently loaded model names.
        
        Returns:
            List of loaded model names
        """
        return await self.get_loaded_models()
    
    async def is_ready(self) -> bool:
        """
        Check if the repository is initialized and healthy.
        
        Returns:
            True if ready, False otherwise
        """
        return self._is_initialized and await self.is_healthy()
    
    async def chat(self, prompt: str, **gen_kwargs) -> str:
        """
        Send a chat request to the LLM.
        
        Args:
            prompt: The prompt to send
            **gen_kwargs: Generation parameters (temperature, max_tokens, etc.)
        
        Returns:
            The LLM response as string
        """
        if not await self.is_ready():
            raise RuntimeError("Repository not initialized. Call initialize() first.")
        
        response = await self.chat_llm(prompt, **gen_kwargs)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def shutdown(self):
        """Clean shutdown of the repository."""
        try:
            if self._is_initialized:
                await self.stop()
                await self.unload_all_models()
                self._is_initialized = False
                logger.info("Repository shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise