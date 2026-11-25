import json
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, Dict, List, Optional, Union

class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    stop: Optional[List[str]] = None
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    # Orchestrator controls (optional, backwards-compatible, RAM-only image_base64)
    use_orchestrator: bool = True
    agentic_mode: bool = Field(
        default=False,
        description="False = Fast 1-pass pre-fetch orchestrator; True = Autonomous multi-step ReAct agent where LLM chooses and chains DB/Web/Weather tools",
    )
    max_agent_steps: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Maximum autonomous tool-calling hops when agentic_mode=True",
    )
    use_web: Optional[bool] = None
    use_rag: Optional[bool] = None
    image_base64: Optional[str] = None
    user_id: Optional[str] = Field(default=None, description="Tenant user_id for strict ChromaDB & RAM isolation")
    session_id: Optional[str] = Field(default=None, description="Per-user chat session_id for conversation/document scoping")

    @model_validator(mode="before")
    @classmethod
    def _parse_json_form_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict):
            # Clean Swagger UI auto-generated dummy placeholders if left untouched
            if data.get("image_base64") == "string":
                data["image_base64"] = None
            if data.get("user_id") == "string":
                data["user_id"] = None
            if data.get("session_id") == "string":
                data["session_id"] = None
            if data.get("tool_choice") == "string":
                data["tool_choice"] = None
            if data.get("stop") == ["string"]:
                data["stop"] = None
            if data.get("response_format") == {"additionalProp1": {}}:
                data["response_format"] = None
            if data.get("stream_options") == {"additionalProp1": {}}:
                data["stream_options"] = None
            if data.get("tools") == [{"additionalProp1": {}}]:
                data["tools"] = None
        return data

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "glm-4-9b-chat-IQ4_XS.gguf",
                "user_id": None,
                "session_id": None,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is the latest news about Nvidia CEO Jensen Huang?"},
                ],
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 512,
                "repeat_penalty": 1.1,
                "stream": False,
                "use_orchestrator": True,
                "agentic_mode": False,
                "max_agent_steps": 3,
                "use_web": True,
                "use_rag": False,
            }
        }
    )

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    choices: List[ChatChoice]
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    agentic_mode: bool = False
    agent_trace: Optional[List[Dict[str, Any]]] = None
    citations: Optional[List[Dict[str, Any]]] = None