from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    stop: Optional[List[str]] = None
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0)
    # Orchestrator controls (optional, backwards-compatible, RAM-only image_base64)
    use_orchestrator: bool = True
    use_web: Optional[bool] = None
    use_rag: Optional[bool] = None
    image_base64: Optional[str] = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}