# App/repository/llmRepository.py
"""
LLM Repository Layer
====================
Provides pure data & engine access for LLM operations.
Wraps the low-level AsyncLLM inference client.
No HTTP exception raising or business rules belong here (handled in LlmService).
"""

from App.core.AsyncLLM import AsyncLLM
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


# Default singleton repository instance
llm_repo = llmChatRepo()
