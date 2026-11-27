# App/repository/llmRepository.py
"""
LLM Repository Layer
====================
Provides pure data & engine access for LLM operations.
Wraps the low-level AsyncLLM inference client.
No HTTP exception raising or business rules belong here (handled in LlmService).
"""

from App.core.AsyncLLM import AsyncLLM
from App.core.settings import settings
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class llmChatRepo(AsyncLLM):
    """
    Engine Access Layer (Repository).
    Interacts with llama-cpp / AsyncLLM execution workers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_initialized = False
        self._model_name = kwargs.get("model_name", None)


# Single shared application-wide repository instance configured from .env settings
llm_repo = llmChatRepo(
    model_path=settings.LLM_MODEL_PATH,
    n_gpu_layers=settings.LLM_N_GPU_LAYERS,
    n_ctx=settings.LLM_N_CTX,
    n_batch=settings.LLM_N_BATCH,
    n_threads=settings.LLM_N_THREADS,
    verbose=settings.LLM_VERBOSE,
    use_mmap=settings.USE_MMAP,
    use_mlock=settings.USE_MLOCK,
    continuous_batch_slots=settings.LLM_CONTINUOUS_BATCH_SLOTS,
    available_ram=settings.LLM_MIN_AVAILABLE_RAM_MB,
)
