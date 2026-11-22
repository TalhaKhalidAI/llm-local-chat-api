import llama_cpp
from pydantic import BaseModel, Field,field_validator,ConfigDict,ValidationError
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings
import gc
import os
from uuid import uuid4
import asyncio
from typing import Dict, List, Union, Any, Optional, Tuple, Type, TypeVar, Generic
import psutil
import json
import re
from enum import Enum
from dataclasses import dataclass,field
import time
import logging
from App.core.settings import Settings
logger = logging.getLogger(__name__)

class EcharrParsers:
    def __init__(self):
        self.cj: Dict = {}
        cdir = os.getcwd()
        # Fix: Initialize as an empty list, not the type 'List[str]'
        self.all_charts_list: List[str] = [] 
        fp = os.path.join(cdir, "Charts.json")
        if not os.path.exists(fp):
            raise ValueError("Error file not found Charts.json")
        try:
            with open(fp, 'r') as f:
                self.cj = json.load(f)
            self.all_charts_list = list(self.cj.get("templates", {}).keys())
        except json.JSONDecodeError as je:
            raise ValueError(f"Error invalid json  in echart{je}")
            # 2. Extract the keys into the list
        
        # 3. Create the Enum LAST using the populated list
        # Using the Functional API (Enum) or the type() method you liked:
        self.DEC = Enum('DEC', {key.upper(): key for key in self.all_charts_list})
        """
        Returns a string describing available charts.
        This is what you 'feed' the LLM so it knows its powers.
        """
        metadata = []
        templates = self.cj.get("templates", {})
        for name, info in templates.items():
            desc = info.get("description", "No description")
            metadata.append(f"- {name}: {desc}")
       

    def get_enum_list(self, capitalise: bool = False):
        """
        Returns list of chart names.
        If capitalise=True: returns uppercase names (enum member names)
        If capitalise=False: returns lowercase values (original keys)
        """
        if capitalise:
            # Return enum member names (already uppercase)
            return [member.name for member in self.DEC]
        else:
            # Return enum member values (original lowercase keys)
            return [member.value for member in self.DEC]
        
    def get_template_placeholders(self, chart_name:'DEC') -> List[str]:
        """Returns the specific blanks the LLM needs to fill"""
        template = self.cj.get("templates", {}).get(chart_name)
        return template.get("placeholders", []) if template else []
    
    def get_full_chart_template(self, chart_name: 'DEC') -> Dict[str, Any]:
        """
        Returns the ENTIRE JSON block for a specific chart.
        """
        # ✅ Correct: Use chart_name.value to get the string key
        chart_key = chart_name.value if hasattr(chart_name, 'value') else chart_name
        template = self.cj.get("templates", {}).get(chart_key)
        
        if not template:
            # ✅ Correct formatting
            raise ValueError(f"Chart '{chart_key}' not found. Available charts: {list(self.cj.get('templates', {}).keys())}")
        
        return template
    
    def get_dynamic_prompt(self, user_input: str, selected_chart:'DEC'):
            """
            Injected Prompt: 
            Only tells the LLM about the ONE chart it needs to fill.
            """
            # 1. Get the specific placeholders for the chosen chart
            placeholders = self.get_template_placeholders(selected_chart)
            
            # 2. Create a string representation of the keys the LLM must return
            # e.g., '"xAxisData": [...], "seriesData": [...]'
            placeholder_str = ", ".join([f'"{p}": [...]' for p in placeholders])

            return f"""
            You are a Data Extraction Bot for the chart: {selected_chart}.
            
            REQUIRED KEYS:
            You must return a JSON object with exactly these keys: {placeholders}
            
            USER DATA:
            {user_input}
            
            STRICT OUTPUT FORMAT:
            {{
                "chart_name": "{selected_chart}",
                "data": {{ {placeholder_str} }}
            }}
            """
    def get_full_injection_prompt(self, user_input: str, selected_chart: 'DEC'):
                full_template = self.get_full_chart_template(selected_chart)
                
                return f"""
                You are a JSON Engineer.
                
                TEMPLATE:
                {json.dumps(full_template["options"])}
                
                USER DATA:
                {user_input}
                
                CRITICAL RULES:
                1. Return the FULL JSON exactly as provided in the TEMPLATE.
                2. ONLY replace the "{{{{placeholders}}}}" with values from User Data.
                3. DO NOT add new keys like 'title', 'tooltip', 'legend', or 'color' if they are not in the template.
                4. If data for a placeholder is missing, return an empty list [].
                """
    
    def get_designer_prompt(self, user_input: str, selected_chart: 'DEC'):
            full_template = self.get_full_chart_template(selected_chart)
            
            return f"""
            You are a Senior Chart Designer for an ERP system.
            
            TEMPLATE:
            {json.dumps(full_template["options"])}
            
            USER DATA: {user_input}
            
            DESIGN RULES:
            1. Fill the "{{{{placeholders}}}}" with the correct data.
            2. If the user mentions a specific TITLE, add/update the "title": {{"text": "..."}} key.
            3. If the user mentions a COLOR (e.g., 'make it red'), update the "itemStyle": {{"color": "..."}} inside the series.
            4. If the user asks for a 'smooth' line, set "smooth": true in the series.
            
            STRICT REQUIREMENT: 
            Only modify keys that exist in the ECharts standard. Do not invent custom keys.
            """

class Runnable:
    def __init__(self, func):
        self.func = func
    
    async def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    
    # ADD THESE METHODS:
    def invoke(self, input_data):
        """Sync invoke"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input_data))
        else:
            return loop.run_until_complete(self.ainvoke(input_data))
    
    async def ainvoke(self, input_data):
        """Async invoke"""
        return await self(input_data)
    
    # def __or__(self, other):
    #     async def chained(value):
    #         result = await self.ainvoke(value)  # CHANGE: use ainvoke
    #         if hasattr(other, 'ainvoke'):
    #             return await other.ainvoke(result)
    #         elif hasattr(other, 'invoke'):
    #             return other.invoke(result)
    #         elif callable(other):
    #             return await other(result) if asyncio.iscoroutinefunction(other) else other(result)
    #         return result
    #     return Runnable(chained)
    def __or__(self, other):
        async def chained(value):
            result = await self.ainvoke(value)
            if asyncio.iscoroutinefunction(other):
                return await other(result)
            elif callable(other):
                return other(result)
            return result
        return Runnable(chained)
 

class StrOutputParser:
    def parse(self, response):
        # Extract content if response is an AIMessage
        if hasattr(response, "content"):
            response = response.content
            
        if isinstance(response, str):
            return response.strip()
        
        if isinstance(response, dict):
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "text" in choice:
                    return choice.get("text", "").strip()
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
            if "text" in response:
                return response["text"].strip()
            return json.dumps(response)
        
        return str(response).strip()
    
    # ADD these methods for consistency
    def invoke(self, input_data):
        return self.parse(input_data)
    
    async def ainvoke(self, input_data):
        return self.parse(input_data)
    
    def __call__(self, response):
        return self.parse(response)
    
    def __or__(self, other):
        def chained(value):
            parsed = self.parse(value)
            return other(parsed) if callable(other) else parsed
        return Runnable(chained)
    

# class Settings(BaseSettings):
#     model_config = ConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         case_sensitive=True,
#         extra="allow"   
#     )
#     MODEL_PATH:str=Field(...)

#     @field_validator("MODEL_PATH",mode="before")
#     @classmethod
#     def validate_path(cls,v:str)->str:
#         if not os.path.exists(v):
#             raise ValueError(f"Error path not found {v}")
#         return v

class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str] = None,
                 partial_variables: Dict[str, Any] = None,
                 validate_template: bool = True):
        self.template = template
        self.partial_variables = partial_variables or {}
        self.input_variables = input_variables or self._extract_variables(template)
        self.validate_template = validate_template
        if validate_template:
            self._validate()

    def _extract_variables(self, template: str) -> List[str]:
        
        return re.findall(r'\{([^{}]+)\}', template)

    def _validate(self):
        """Check that all input variables are accounted for."""
        pass

    def format(self, **kwargs) -> str:
        """Combine partial and passed variables."""
        all_vars = {**self.partial_variables, **kwargs}
        if self.validate_template:
            missing = set(self.input_variables) - set(all_vars)
            if missing:
                raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**all_vars)

    def partial(self, **kwargs):
        """Return a new template with pre‑filled variables."""
        new_partials = {**self.partial_variables, **kwargs}
        return PromptTemplate(
            template=self.template,
            partial_variables=new_partials,
            validate_template=self.validate_template
        )

    def invoke(self, input_data):
        """Sync invoke - format prompt"""
        if isinstance(input_data, dict):
            return self.format(**input_data)
        return self.format(input=input_data)
    
    async def ainvoke(self, input_data):
        """Async invoke - format prompt"""
        return self.invoke(input_data)
    
    # def __or__(self, other):
    #     async def chained(input_data):
    #         if isinstance(input_data, dict):
    #             prompt = self.format(**input_data)
    #         else:
    #             prompt = self.format(input=input_data)
            
    #         if hasattr(other, 'ainvoke'):
    #             return await other.ainvoke(prompt)
    #         elif hasattr(other, 'invoke'):
    #             return other.invoke(prompt)
    #         elif callable(other):
    #             return await other(prompt) if asyncio.iscoroutinefunction(other) else other(prompt)
    #         return prompt
    #     return Runnable(chained)

    def __or__(self, other):
        async def chained(input_data):
            if isinstance(input_data, dict):
                # Format prompt
                prompt = self.format(**input_data)
                # Pass through other keys (like chat_id, timeout)
                result = {**input_data, "input": prompt}
            else:
                prompt = self.format(input=input_data)
                result = {"input": prompt}
            
            if hasattr(other, 'ainvoke'):
                return await other.ainvoke(result)
            elif hasattr(other, 'invoke'):
                return other.invoke(result)
            elif callable(other):
                return await other(result) if asyncio.iscoroutinefunction(other) else other(result)
            return result
        return Runnable(chained)
    
    @classmethod
    def from_template(cls, template: str):
        """Convenience constructor."""
        return cls(template)

class ChatPromptTemplate:

    def __init__(self, messages: List[Tuple[str, str]]):
        """# Store the messages for later formatting"""
        self.messages = messages

    def format_prompt(self, **kwargs):
        """Replace placeholders and return prompt string"""
        result = ""
        
        for role, template in self.messages:
            # Replace placeholders like {name} with values
            content = template.format(**kwargs)
            
            # Convert role to format LLM understands
            if role == "system":
                result += f"[SYSTEM] {content}\n"
            elif role == "human":
                result += f"[USER] {content}\n"
            elif role == "assistant":
                result += f"[ASSISTANT] {content}\n"
        
        # Add final marker for LLM to start generating
        result += "[ASSISTANT] "
        return result
    def __call__(self, **kwargs):
        """Allow: template(topic="Python")"""
        return self.format_prompt(**kwargs)
        
    # def __or__(self, other):
    #     async def chained(input_data):
    #         if isinstance(input_data, dict):
    #             prompt = self.format_prompt(**input_data)
    #         else:
    #             prompt = self.format_prompt(input=input_data)
    #         return await other(prompt)   # <-- always await
    #     return Runnable(chained)

    def __or__(self, other):
        """Chain with metadata preservation (keeps chat_id, timeout, etc.)"""
        async def chained(input_data):
            if isinstance(input_data, dict):
                # Format the prompt using all values from the dict
                prompt = self.format_prompt(**input_data)
                # ✅ PRESERVE all original data + add formatted prompt
                result = {**input_data, "input": prompt}
            else:
                # Input is a string, treat as user input
                prompt = self.format_prompt(input=input_data)
                result = {"input": prompt}
            
            # Pass to next component with metadata preserved
            if hasattr(other, 'ainvoke'):
                return await other.ainvoke(result)
            elif hasattr(other, 'invoke'):
                return other.invoke(result)
            elif callable(other):
                return await other(result) if asyncio.iscoroutinefunction(other) else other(result)
            return result
        return Runnable(chained)
    @classmethod
    def from_messages(cls, messages:List[Tuple[str,str]]):
        """Create template from message list"""
        return cls(messages)

    @classmethod
    def from_template(cls, template: str):
        """Create simple template from one string"""
        return cls([("human", template)])

class AIMessage(BaseModel):
    """Message from an AI."""
    content: str
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __str__(self):
        return self.content
    
    def __getitem__(self, key):
        """Allow dict-like access for backward compatibility"""
        return getattr(self, key) if hasattr(self, key) else self.response_metadata.get(key)

T = TypeVar('T', bound=BaseModel)

class JsonOutputParser:
    """
    LangChain-style JSON Output Parser.
    Extracts JSON from LLM responses, handles markdown code blocks, and returns dict.
    """
    
    def __init__(self, pydantic_object: Optional[Type[BaseModel]] = None):
        """
        Args:
            pydantic_object: Optional Pydantic model for validation.
                            If provided, returns validated model instead of dict.
        """
        self.pydantic_object = pydantic_object
    
    def parse(self, response):
        """Parse raw LLM response into JSON/dict"""
        import re
        # Extract content if response is an AIMessage
        if hasattr(response, "content"):
            response = response.content

        if isinstance(response, str):
            text = response.strip()
            
            # Remove markdown code blocks
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    text = json_match.group()
            
            # Parse JSON
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}\nResponse: {text[:200]}")
            
            # Validate with Pydantic if provided
            if self.pydantic_object:
                try:
                    return self.pydantic_object(**parsed)
                except PydanticValidationError as e:
                    raise ValueError(f"Pydantic validation failed: {e}")
            return parsed
        
        elif isinstance(response, dict):
            if self.pydantic_object:
                return self.pydantic_object(**response)
            return response
        
        return response
    
    def get_format_instructions(self) -> str:
        """Return formatting instructions for the LLM - similar to LangChain's [citation:5]"""
        if self.pydantic_object:
            schema = self.pydantic_object.model_json_schema()
            return f"""Respond with a valid JSON object matching this schema:
{json.dumps(schema, indent=2)}"""
        return "Respond with a valid JSON object. Do not include any explanatory text."
    
    def invoke(self, input_data):
        return self.parse(input_data)
    
    async def ainvoke(self, input_data):
        return self.parse(input_data)
    
    def __call__(self, response):
        return self.parse(response)

@dataclass(slots=True)  # ✅ slots=True reduces memory usage (~40% less)
class Document:
    """
    Optimized LangChain-style Document class for RAG pipelines.
    
    Performance Optimizations:
    - __slots__ reduces memory footprint by ~40%
    - Lazy JSON serialization (only when needed)
    - Efficient string concatenation with join
    - Early returns for common operations
    - Minimal overhead for metadata access
    """
    
    # Core fields with type hints
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    # Class-level constants
    _TRUNCATE_SUFFIX = "..."
    _MAX_PREVIEW_LEN = 50

    def __post_init__(self):
        """Auto-generate ID if not provided (minimal overhead)"""
        if self.id is None:
            # ✅ Use faster UUID generation for small IDs
            self.id = uuid4().hex[:8]
    
    def __str__(self) -> str:
        """Fast string conversion"""
        return self.page_content
    
    def __repr__(self) -> str:
        """Optimized debug representation"""
        content = self.page_content
        if len(content) > self._MAX_PREVIEW_LEN:
            content = content[:self._MAX_PREVIEW_LEN] + self._TRUNCATE_SUFFIX
        return f"Document(id='{self.id}', page_content='{content}', metadata={self.metadata})"
    
    def __len__(self) -> int:
        """O(1) length operation"""
        return len(self.page_content)
    
    def __bool__(self) -> bool:
        """Truthy if has content"""
        return bool(self.page_content)
    
    def __eq__(self, other: object) -> bool:
        """Equality check by ID"""
        if not isinstance(other, Document):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID for dict/set usage"""
        return hash(self.id)
    
    def __add__(self, other: 'Document') -> 'Document':
        """Fast document combination"""
        if not isinstance(other, Document):
            return NotImplemented
        
        # ✅ Use efficient string concatenation
        return Document(
            page_content=f"{self.page_content}\n{other.page_content}",
            metadata={
                "sources": [self.metadata, other.metadata],
                "original_ids": [self.id, other.id]
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with minimal copying"""
        return {
            "id": self.id,
            "page_content": self.page_content,
            "metadata": dict(self.metadata)  # Copy to prevent modification
        }
    
    def to_json(self, indent: int = None, compact: bool = False) -> str:
        """
        Convert to JSON with options for performance.
        
        Args:
            indent: Pretty print indent (None = compact)
            compact: Force compact JSON even with indent (ignores indent)
        """
        if compact or indent is None:
            # ✅ Fastest JSON generation (no whitespace)
            return json.dumps(self.to_dict(), separators=(',', ':'), ensure_ascii=False)
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Fast dict creation with defaults"""
        return cls(
            page_content=data.get("page_content", ""),
            metadata=data.get("metadata", {}),
            id=data.get("id")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Document':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def combine(cls, documents: List['Document'], separator: str = "\n") -> 'Document':
        """
        Optimized combine using list comprehension and efficient join.
        """
        if not documents:
            raise ValueError("Cannot combine empty list of documents")
        
        # ✅ Single pass through documents for content and metadata
        contents = []
        original_ids = []
        sources = []
        
        for doc in documents:
            contents.append(doc.page_content)
            original_ids.append(doc.id)
            sources.append(doc.metadata)
        
        # ✅ Efficient string joining
        combined_content = separator.join(contents)
        
        return cls(
            page_content=combined_content,
            metadata={
                "combined": True,
                "original_count": len(documents),
                "original_ids": original_ids,
                "sources": sources
            }
        )
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Fast metadata access with default"""
        return self.metadata.get(key, default)
    
    def has_metadata(self, key: str) -> bool:
        """Fast metadata existence check"""
        return key in self.metadata
    
    def add_metadata(self, key: str, value: Any) -> 'Document':
        """
        Add metadata and return new document (immutable).
        Optimized to avoid copying large metadata dict when possible.
        """
        # ✅ Create new dict with single copy operation
        new_metadata = {**self.metadata, key: value}
        return Document(
            page_content=self.page_content,
            metadata=new_metadata,
            id=self.id
        )
    
    def truncate(self, max_chars: int = 500, suffix: str = "...") -> 'Document':
        """
        Optimized truncation with early return.
        """
        # ✅ Early return if no truncation needed
        if len(self.page_content) <= max_chars:
            return self
        
        # ✅ Calculate truncation point
        trunc_len = max_chars - len(suffix)
        if trunc_len <= 0:
            # If max_chars is too small, return just the suffix
            truncated_content = suffix
        else:
            truncated_content = self.page_content[:trunc_len] + suffix
        
        # ✅ Only add truncation metadata if changed
        return Document(
            page_content=truncated_content,
            metadata={
                **self.metadata,
                "truncated": True,
                "original_length": len(self.page_content),
                "truncated_length": max_chars
            },
            id=self.id
        )
    
    def copy(self, **kwargs) -> 'Document':
        """
        Create a copy with optional field updates.
        Optimized for common use cases.
        """
        return Document(
            page_content=kwargs.get("page_content", self.page_content),
            metadata=kwargs.get("metadata", self.metadata.copy()),
            id=kwargs.get("id", self.id)
        )
    
    def batch_to_dicts(self, documents: List['Document']) -> List[Dict[str, Any]]:
        """Convert multiple documents to dicts efficiently"""
        return [doc.to_dict() for doc in documents]
    
    @staticmethod
    def batch_from_dicts(dicts: List[Dict[str, Any]]) -> List['Document']:
        """Create multiple documents from dicts efficiently"""
        return [Document.from_dict(d) for d in dicts]
 
class PydanticOutputParser(JsonOutputParser, Generic[T]):
    """
    LangChain-style Pydantic Output Parser.
    Parses LLM output directly into a Pydantic model [citation:5].
    """
    
    def __init__(self, pydantic_object: Type[T]):
        super().__init__(pydantic_object=pydantic_object)
        self.pydantic_object = pydantic_object
    
    def parse(self, response):
        """Parse and return validated Pydantic model"""
        return super().parse(response)



class AsyncLLM:
    class Config:
        """Central configuration for AsyncLLM."""
        def __init__(self, **kwargs):
            self.available_ram = kwargs.get("available_ram", 3000)          # MB
            self.queue_maxsize = kwargs.get("queue_maxsize", 10)
            self.default_timeout = kwargs.get("default_timeout", 160.0)   # seconds
            self.graceful_shutdown = kwargs.get("graceful_shutdown", True)
            self.enable_metrics = kwargs.get("enable_metrics", True)
            self.enable_cancellation = kwargs.get("enable_cancellation", True)
            self.reject_on_full_queue = kwargs.get("reject_on_full_queue", True)

    def __init__(self, **kwargs):
        """
            Initialize AsyncLLM with configurable parameters.
            
            Kwargs:
                **model_path:** Path to directory containing .gguf model files.
                **temperature:** Sampling temperature (0.0-2.0). Lower = more deterministic.
                **top_p:** Nucleus sampling probability (0.0-1.0).
                **top_k:** Top-k sampling (0 = disabled).
                **n_predict:** Maximum tokens to generate.
                **n_gpu_layers:** GPU offload layers (-1=auto, 0=CPU only).
                **n_ctx:** Context window size in tokens.
                **n_threads:** Number of CPU threads for inference.
                **n_batch:** Batch size for prompt processing.
                **streaming:** Enable streaming by default.
                **repeat_penalty:** Penalty for repeating tokens.
                **verbose:** Enable llama.cpp verbose logging.
                **stop:** List of stop tokens.
                **use_mmap:** Enable mmap for model loading in llama.cpp (default: True).
                **use_mlock:** Lock model in memory in llama.cpp (default: False).
                **
                **queue_maxsize:** Maximum queue capacity.
                **default_timeout:** Request timeout in seconds.
                **auto_restart_worker:** Auto-restart worker on crash.
                **worker_exception_callback:** Function called on worker crash.
                **reject_on_full_queue:** Raise QueueFull instead of waiting.
                **graceful_shutdown:** Wait for pending requests on shutdown.
                **enable_metrics:** Collect request metrics.
                **enable_cancellation:** Allow cancelling in-flight requests.
                **available_ram:** Minimum RAM required (MB) before loading model.
            """
        # Load configuration
        self.setting = Settings()
        config_values = {
            "default_timeout": self.setting.DEFAULT_TIMEOUT,
            "available_ram": self.setting.LLM_MIN_AVAILABLE_RAM_MB,
            **kwargs,
        }
        self.config = self.Config(**config_values)

        # Setup logging
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        self._metrics_lock = asyncio.Lock()
        self.model_path = kwargs.get("model_path")
        self._worker_exception_callback = kwargs.get("worker_exception_callback", None)
        self._auto_restart_worker = kwargs.get("auto_restart_worker", False)
        self._stream_cancel_flags: Dict[str, asyncio.Event] = {}
        self.futures: Dict[str, asyncio.Future] = {}
        self._active_streams: set = set()
        self.current_model: Dict[str, llama_cpp.Llama] = {}
        
        # Metrics
        self._metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "streaming_requests": 0,
            "completed_requests": 0,
        }
        if self.model_path:
            self.path = self.model_path
        else:
            try:
                self.path = getattr(self.setting, "MODEL_PATH", None) or getattr(self.setting, "LLM_MODEL_PATH", None)
                if not self.path:
                    raise ValueError("MODEL_PATH not set.")
            except Exception:
                raise ValueError("MODEL_PATH not set. Provide it via .env or pass model_path argument.")
        

        self.waiting_queue = asyncio.Queue(maxsize=self.config.queue_maxsize)
        self.queue = self.waiting_queue  # Backwards-compatible alias
        self._running_event = asyncio.Event()
        self._worker_task = None
        self._shutdown_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self._future_lock = asyncio.Lock()
        # Infrastructure / Model Loading (Hardware allocation at startup)
        self.n_batch = kwargs.get("n_batch", getattr(self.setting, "LLM_N_BATCH", 512))
        self.n_ctx = kwargs.get("n_ctx", getattr(self.setting, "LLM_N_CTX", 16384)) 
        self.n_threads = kwargs.get("n_threads", getattr(self.setting, "LLM_N_THREADS", 6))
        self.n_gpu_layers = kwargs.get("n_gpu_layers", getattr(self.setting, "LLM_N_GPU_LAYERS", -1))
        self.verbose = kwargs.get("verbose", getattr(self.setting, "LLM_VERBOSE", False))
        self.use_mmap = kwargs.get("use_mmap", getattr(self.setting, "USE_MMAP", True))
        self.use_mlock = kwargs.get("use_mlock", getattr(self.setting, "USE_MLOCK", False))

        # Continuous Batching Slots & Scheduler State
        self.continuous_batch_slots = int(kwargs.get("continuous_batch_slots", getattr(self.setting, "LLM_CONTINUOUS_BATCH_SLOTS", 2)))
        self._model_slots: Dict[str, List[llama_cpp.Llama]] = {}
        self._available_slots: asyncio.Queue = asyncio.Queue()
        self._active_slot_tasks: Dict[str, asyncio.Task] = {}

        # Default fallback generation parameters (used when API caller omits them)
        self.temperature = kwargs.get("temperature", getattr(self.setting, "DEFAULT_TEMPERATURE", 0.7))
        self.top_p = kwargs.get("top_p", getattr(self.setting, "DEFAULT_TOP_P", 0.9))
        self.top_k = kwargs.get("top_k", getattr(self.setting, "DEFAULT_TOP_K", 40))
        self.streaming = kwargs.get("streaming", False)
        self.repeat_penalty = kwargs.get("repeat_penalty", getattr(self.setting, "DEFAULT_REPEAT_PENALTY", 1.1))
        self.n_predict = kwargs.get("n_predict", getattr(self.setting, "DEFAULT_MAX_TOKENS", 2048))
        self.stops = kwargs.get(
            "stop",
            [
                "<|endoftext|>",
                "<|im_end|>",
                "<|user|>",
                "<|observation|>",
                "<|eot_id|>",
                "<|end_of_text|>",
                "<|end_of_turn|>",
                "<end_of_turn>",
                "<eos>",
                "<|end|>",
                "<｜end▁of▁sentence｜>",
                "<｜User｜>",
                "</s>",
            ],
        )
        self._stream_lock = asyncio.Lock()
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        self._last_usage: Dict[str, Dict[str, int]] = {}
        # ✅ FIXED: Added missing attributes
        self._active_requests = 0
        self._unload_allowed = asyncio.Event()
        self._unload_allowed.set()
        self._model_lock = asyncio.Lock()

    # Exact per-model stop token map for all GGUF models in /mnt/llm_models/LLM_Models/
    MODEL_STOP_TOKENS: Dict[str, List[str]] = {
        "deepseek-coder-7b-instruct-v1.5-Q3_K_L.gguf": [
            "<|EOT|>",
            "<｜end▁of▁sentence｜>",
            "### Instruction:",
            "<|endoftext|>",
            "</s>",
        ],
        "deepseek-llm-7b-chat.Q4_K_M.gguf.1": [
            "<｜end▁of▁sentence｜>",
            "User:",
            "<|endoftext|>",
            "</s>",
        ],
        "deepseek-v2-lite-chat-q4_k_m.gguf": [
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "User:",
            "<|endoftext|>",
            "</s>",
        ],
        "gemma-2-9b-it-Q4_K_S.gguf": [
            "<end_of_turn>",
            "<eos>",
            "<start_of_turn>user",
            "<|endoftext|>",
        ],
        "gemma-2-9b-it-Q5_K_M.gguf": [
            "<end_of_turn>",
            "<eos>",
            "<start_of_turn>user",
            "<|endoftext|>",
        ],
        "glm-4-9b-chat-IQ4_XS.gguf": [
            "<|endoftext|>",
            "<|user|>",
            "<|observation|>",
        ],
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": [
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|endoftext|>",
        ],
        "openhermes-2.5-mistral-7b.Q4_K_S.gguf": [
            "<|im_end|>",
            "<|im_start|>user",
            "</s>",
            "<|endoftext|>",
        ],
        "qwen2.5-coder-7b-instruct-q4_k_m.gguf": [
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>user",
        ],
        "Qwen2-VL-2B-Instruct-Q4_K_M.gguf": [
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>user",
        ],
        "SmolVLM-500M-Instruct-Q8_0.gguf": [
            "<end_of_utterance>",
            "<|im_end|>",
            "<|endoftext|>",
            "User:",
        ],
    }

    def get_stops_for_model(self, model_name: Optional[str] = None, user_stops: Optional[List[str]] = None) -> List[str]:
        """
        Return the exact hardcoded stop tokens for the given model (or loaded model),
        merged with universal fallbacks and any custom `user_stops` passed in the API request.
        """
        target = model_name or (list(self.current_model.keys())[0] if self.current_model else "")
        exact = list(self.MODEL_STOP_TOKENS.get(target, []))
        if not exact and target:
            low = target.lower()
            if "glm-4" in low or "glm4" in low:
                exact = ["<|endoftext|>", "<|user|>", "<|observation|>"]
            elif "gemma" in low:
                exact = ["<end_of_turn>", "<eos>", "<start_of_turn>user"]
            elif "llama-3" in low or "llama3" in low:
                exact = ["<|eot_id|>", "<|end_of_text|>"]
            elif "deepseek-coder" in low:
                exact = ["<|EOT|>", "<｜end▁of▁sentence｜>", "### Instruction:"]
            elif "deepseek" in low:
                exact = ["<｜end▁of▁sentence｜>", "<｜User｜>", "User:"]
            elif "qwen" in low or "openhermes" in low:
                exact = ["<|im_end|>", "<|endoftext|>"]
            elif "smolvlm" in low:
                exact = ["<end_of_utterance>", "<|im_end|>", "<|endoftext|>"]

        merged: List[str] = []
        for tok in exact + list(self.stops) + (list(user_stops) if user_stops else []):
            if tok and tok not in merged:
                merged.append(tok)
        return merged
    def pop_last_usage(self, task_id: str) -> Dict[str, int]:
        return self._last_usage.pop(task_id, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    # ------------------------ Health & Metrics ------------------------
    async def is_healthy(self) -> bool:
        async with self.lock:
            return self._running_event.is_set() and len(self.current_model) > 0

    async def get_metrics(self) -> Dict[str, Any]:
        async with self._metrics_lock:
            metrics = self._metrics.copy()
            metrics["waiting_queue_size"] = self.waiting_queue.qsize()
            metrics["queue_size"] = self.waiting_queue.qsize()
            metrics["active_batch_slots"] = self.continuous_batch_slots - self._available_slots.qsize()
            metrics["available_slots"] = self._available_slots.qsize()
            metrics["total_batch_slots"] = self.continuous_batch_slots
            metrics["active_streams"] = len(self._active_streams)
            metrics["models_loaded"] = len(self.current_model)
            return metrics

    # ------------------------ Request Cancellation ------------------------
    async def cancel_request(self, request_id: str):
        async with self._future_lock:
            if request_id in self.futures and not self.futures[request_id].done():
                self.futures[request_id].set_exception(asyncio.CancelledError(f"Request {request_id} cancelled"))
                logger.info(f"Cancelled request {request_id}")
        
        if request_id in self._stream_cancel_flags:
            self._stream_cancel_flags[request_id].set()
            logger.info(f"Signalled cancel to stream {request_id}")
            
        if request_id in self._active_slot_tasks:
            self._active_slot_tasks[request_id].cancel()
            logger.info(f"Cancelled active slot task for request {request_id}")
        
        logger.warning(f"Request {request_id} not found")
    # ------------------------ Model Management ------------------------
    async def _get_model_in_current_dirs(self):
        try:
            return await asyncio.to_thread(self._walk_directory)
        except Exception as e:
            raise ValueError(f"Error getting models: {e}")

    def _walk_directory(self):
        all_files = []
        for root, _, files in os.walk(self.path):
            for f in files:
                if f.endswith(".gguf"):
                    bp = os.path.join(root, f)
                    all_files.append({"path": bp, "filename": f, "directory": root})
        return all_files

    async def _load_model(self, **kwargs: dict):
        async with self.lock:
            try:
                model_name = kwargs.get("model_name")
                if not model_name:
                    raise ValueError("model_name is required")
                avail = psutil.virtual_memory().available / (1024**2)
                if avail < self.config.available_ram:
                    raise MemoryError(f"Need {self.config.available_ram/1000}GB free, have {avail:.0f}MB")

                # Hardware / Infrastructure parameters for loading model
                self.n_batch = kwargs.get("n_batch", self.n_batch)
                self.n_ctx = kwargs.get("n_ctx", self.n_ctx)
                self.n_threads = kwargs.get("n_threads", self.n_threads)
                self.n_gpu_layers = kwargs.get("n_gpu_layers", self.n_gpu_layers)
                self.verbose = kwargs.get("verbose", self.verbose)
                self.use_mmap = kwargs.get("use_mmap", self.use_mmap)
                self.use_mlock = kwargs.get("use_mlock", self.use_mlock)

                if model_name in self.current_model:
                    raise ValueError(f"Model '{model_name}' already loaded")

                fmp = os.path.join(self.path, model_name)
                if not os.path.exists(fmp):
                    models = await self._get_model_in_current_dirs()
                    found = next((m for m in models if m["filename"] == model_name), None)
                    if found:
                        fmp = found["path"]
                    else:
                        raise FileNotFoundError(f"Model '{model_name}' not found in '{self.path}'")

                logger.info(
                    f"Loading single shared model for True Continuous Batching: {fmp} ("
                    f"n_ctx={self.n_ctx}, "
                    f"n_seq_max={self.continuous_batch_slots}, "
                    f"n_gpu_layers={self.n_gpu_layers}, "
                    f"n_batch={self.n_batch}, "
                    f"n_threads={self.n_threads}, "
                    f"use_mmap={self.use_mmap}, "
                    f"use_mlock={self.use_mlock})"
                )
                def _load_shared_model():
                    m = llama_cpp.Llama(
                        model_path=fmp,
                        n_batch=self.n_batch,
                        n_ctx=self.n_ctx,
                        n_threads=self.n_threads,
                        n_gpu_layers=self.n_gpu_layers,
                        verbose=self.verbose,
                        use_mmap=self.use_mmap,
                        use_mlock=self.use_mlock,
                    )
                    if self.continuous_batch_slots > 1:
                        from llama_cpp._internals import LlamaContext
                        cparams = m.context_params
                        cparams.n_seq_max = self.continuous_batch_slots
                        cparams.n_ctx = self.n_ctx
                        m._ctx.close()
                        m._ctx = LlamaContext(model=m._model, params=cparams)
                    return m

                llm = await asyncio.to_thread(_load_shared_model)
                self.current_model[model_name] = llm
                self._model_slots[model_name] = [llm]

                # Populate available continuous batching sequence IDs (0 .. n_seq_max - 1)
                while not self._available_slots.empty():
                    try:
                        self._available_slots.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                for s_idx in range(self.continuous_batch_slots):
                    await self._available_slots.put(s_idx)

                logger.info(
                    f"Model '{model_name}' loaded ONCE with {self.continuous_batch_slots} shared KV sequence slot(s)"
                )
                return llm
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    # ------------------------ Worker Lifecycle ------------------------
    def _worker_done_callback(self, task):
        if task.cancelled():
            logger.warning("Worker task cancelled")
            return
        exception = task.exception()
        if exception:
            logger.error(f"Worker crashed: {exception}", exc_info=True)
            if self._worker_exception_callback:
                try:
                    self._worker_exception_callback(exception)
                except Exception as e:
                    logger.error(f"Exception callback failed: {e}")
            if self._auto_restart_worker:
                logger.info("Auto‑restarting worker...")
                self._worker_task = asyncio.create_task(self._worker())
                self._worker_task.add_done_callback(self._worker_done_callback)

    async def format_messages_native(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Format messages using the loaded GGUF model's native Jinja2 chat_template.
        Falls back cleanly to role-tagged prompt if no template is present.
        """
        async with self.lock:
            model = list(self.current_model.values())[0] if self.current_model else None

        if model is not None:
            meta = getattr(model, "metadata", {}) or {}
            template_str = meta.get("tokenizer.chat_template")
            if template_str:
                try:
                    import jinja2

                    vocab = llama_cpp.llama_model_get_vocab(model.model)
                    bos_id = llama_cpp.llama_vocab_bos(vocab)
                    eos_id = llama_cpp.llama_vocab_eos(vocab)
                    bos_tok = model.detokenize([bos_id]).decode("utf-8", errors="ignore") if bos_id >= 0 else ""
                    eos_tok = model.detokenize([eos_id]).decode("utf-8", errors="ignore") if eos_id >= 0 else ""

                    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
                    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
                    tpl = env.from_string(template_str)
                    render_kwargs: Dict[str, Any] = {
                        "messages": messages,
                        "add_generation_prompt": True,
                        "bos_token": bos_tok,
                        "eos_token": eos_tok,
                    }
                    if tools:
                        render_kwargs["tools"] = tools
                    rendered = tpl.render(**render_kwargs)
                    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                    if last_user and last_user[:40] not in rendered:
                        raise ValueError("Template omitted user content")
                    return rendered
                except Exception as exc:
                    logger.warning(f"Native chat_template render failed ({exc}); using fallback formatting")

        parts = [f"[{m.get('role', 'user').upper()}] {m.get('content', '')}" for m in messages]
        return "\n".join(parts) + "\n[ASSISTANT] "

    @staticmethod
    def _grammar_from_response_format(response_format: Optional[Dict[str, Any]]):
        if not response_format or not isinstance(response_format, dict):
            return None
        rtype = response_format.get("type")
        if rtype == "json_schema":
            schema_obj = response_format.get("json_schema", {})
            schema = schema_obj.get("schema", schema_obj) if isinstance(schema_obj, dict) else schema_obj
            if schema:
                return llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))
        if rtype == "json_object":
            return llama_cpp.LlamaGrammar.from_json_schema(json.dumps({"type": "object"}))
        return None

    def _build_sampler(self, model: llama_cpp.Llama, task: dict):
        """Build a per-sequence llama_sampler chain for continuous batching (with optional GBNF grammar)."""
        temp = task.get("temperature")
        temp = self.temperature if temp is None else float(temp)
        top_k = task.get("top_k")
        top_k = self.top_k if top_k is None else int(top_k)
        top_p = task.get("top_p")
        top_p = self.top_p if top_p is None else float(top_p)
        repeat_penalty = task.get("repeat_penalty")
        repeat_penalty = (self.repeat_penalty if self.repeat_penalty is not None else 1.0) if repeat_penalty is None else float(repeat_penalty)

        sparams = llama_cpp.llama_sampler_chain_default_params()
        chain = llama_cpp.llama_sampler_chain_init(sparams)
        vocab = llama_cpp.llama_model_get_vocab(model.model)
        n_vocab = llama_cpp.llama_vocab_n_tokens(vocab)

        grammar_obj = self._grammar_from_response_format(task.get("response_format"))
        if grammar_obj is not None:
            llama_cpp.llama_sampler_chain_add(
                chain,
                llama_cpp.llama_sampler_init_grammar(
                    vocab,
                    grammar_obj._grammar.encode("utf-8"),
                    grammar_obj._root.encode("utf-8"),
                ),
            )

        llama_cpp.llama_sampler_chain_add(
            chain,
            llama_cpp.llama_sampler_init_penalties(n_vocab, 64, repeat_penalty, 0.0, 0.0),
        )
        if temp <= 0.0:
            llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_greedy())
        else:
            if top_k > 0:
                llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_top_k(top_k))
            if 0.0 < top_p < 1.0:
                llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_top_p(top_p, 1))
            llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_temp(temp))
            llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_dist(int(time.time() * 1000) & 0xFFFFFFFF))
        return chain

    def _step_continuous_batch_sync(self, model: llama_cpp.Llama, active_seqs: Dict[int, dict]) -> List[tuple]:
        """
        Execute ONE true iteration-level continuous batching step across all active sequences.
        Packs chunked prefill tokens + decode tokens into a single `llama_batch`, runs ONE
        `llama_decode()`, and samples one token for every sequence ready for logits.
        """
        ctx = model.ctx
        vocab = llama_cpp.llama_model_get_vocab(model.model)
        n_batch_cap = max(1, int(self.n_batch))
        batch = llama_cpp.llama_batch_init(n_batch_cap, 0, 1)
        events: List[tuple] = []

        try:
            n_tokens = 0
            logit_targets: List[tuple[int, int]] = []  # (batch_idx, seq_id)

            # Pack decode tokens first (1 token each, low latency), then chunked prefill tokens
            for seq_id, seq in active_seqs.items():
                if seq["prefill_idx"] >= len(seq["prompt_tokens"]) and n_tokens < n_batch_cap:
                    b_idx = n_tokens
                    batch.token[b_idx] = seq["next_token"]
                    batch.pos[b_idx] = seq["pos"]
                    batch.n_seq_id[b_idx] = 1
                    batch.seq_id[b_idx][0] = seq_id
                    batch.logits[b_idx] = 1
                    seq["pos"] += 1
                    n_tokens += 1
                    logit_targets.append((b_idx, seq_id))

            for seq_id, seq in active_seqs.items():
                p_tokens = seq["prompt_tokens"]
                p_idx = seq["prefill_idx"]
                while p_idx < len(p_tokens) and n_tokens < n_batch_cap:
                    b_idx = n_tokens
                    is_last_prompt_tok = (p_idx == len(p_tokens) - 1)
                    batch.token[b_idx] = p_tokens[p_idx]
                    batch.pos[b_idx] = seq["pos"]
                    batch.n_seq_id[b_idx] = 1
                    batch.seq_id[b_idx][0] = seq_id
                    batch.logits[b_idx] = 1 if is_last_prompt_tok else 0
                    seq["pos"] += 1
                    p_idx += 1
                    n_tokens += 1
                    if is_last_prompt_tok:
                        logit_targets.append((b_idx, seq_id))
                seq["prefill_idx"] = p_idx

            batch.n_tokens = n_tokens
            if n_tokens == 0:
                return events

            rc = llama_cpp.llama_decode(ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode failed with code {rc}")

            for b_idx, seq_id in logit_targets:
                seq = active_seqs[seq_id]
                tok = llama_cpp.llama_sampler_sample(seq["sampler"], ctx, b_idx)

                is_eog = bool(llama_cpp.llama_vocab_is_eog(vocab, tok))
                seq["generated_count"] += 1

                piece = ""
                if not is_eog:
                    tok_bytes = model.detokenize([tok], special=True)
                    tok_str = tok_bytes.decode("utf-8", errors="ignore").strip()
                    if tok_str in seq["stops"]:
                        is_eog = True
                    else:
                        normal_bytes = model.detokenize([tok])
                        seq["byte_buf"].extend(normal_bytes)
                        # Flush valid UTF-8 prefix safely
                        try:
                            piece = seq["byte_buf"].decode("utf-8")
                            seq["byte_buf"].clear()
                        except UnicodeDecodeError:
                            for cut in range(len(seq["byte_buf"]) - 1, -1, -1):
                                try:
                                    piece = seq["byte_buf"][:cut].decode("utf-8")
                                    del seq["byte_buf"][:cut]
                                    break
                                except UnicodeDecodeError:
                                    continue

                seq["text_buf"] += piece
                seq["next_token"] = tok

                stop_hit = any(s and s in seq["text_buf"] for s in seq["stops"])
                max_hit = seq["generated_count"] >= seq["max_tokens"]
                per_seq_ctx = max(128, self.n_ctx // max(1, self.continuous_batch_slots))
                ctx_hit = seq["pos"] >= per_seq_ctx - 2
                finished = is_eog or stop_hit or max_hit or ctx_hit

                if stop_hit:
                    for s in seq["stops"]:
                        if s and seq["text_buf"].endswith(s):
                            seq["text_buf"] = seq["text_buf"][: -len(s)]
                            piece = ""

                events.append((seq_id, piece, finished, "stop" if (is_eog or stop_hit) else "length"))
            return events
        finally:
            llama_cpp.llama_batch_free(batch)

    def _release_seq_sync(self, model: llama_cpp.Llama, seq_id: int, sampler: Any) -> None:
        if sampler is not None:
            llama_cpp.llama_sampler_free(sampler)
        mem = llama_cpp.llama_get_memory(model.ctx)
        if mem is not None:
            llama_cpp.llama_memory_seq_rm(mem, seq_id, -1, -1)

    async def _worker(self):
        """
        True Iteration-Level Continuous Batching Scheduler.
        Uses ONE shared Llama model + unified KV cache (`n_seq_max=continuous_batch_slots`).
        Admits new requests immediately into freed sequence slots between token steps.
        """
        logger.info(f"True Continuous Batching Scheduler active ({self.continuous_batch_slots} sequence slot(s)).")
        active_seqs: Dict[int, dict] = {}
        model: Optional[llama_cpp.Llama] = None
        pending_task: Optional[dict] = None

        try:
            while self._running_event.is_set():
                # 1. Admit queued requests into any free sequence slots
                while not self._available_slots.empty():
                    if pending_task is not None:
                        task = pending_task
                        pending_task = None
                    else:
                        if not active_seqs and self.waiting_queue.empty():
                            break
                        try:
                            task = self.waiting_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    future = task["future"]
                    if future.cancelled() or future.done():
                        self.waiting_queue.task_done()
                        continue

                    slot_id = await self._available_slots.get()
                    try:
                        if model is None:
                            model = await self._acquire_slot_model(slot_id)
                        else:
                            await self._acquire_slot_model(slot_id)

                        prompt_tokens = await asyncio.to_thread(
                            model.tokenize, task["prompt"].encode("utf-8"), True, True
                        )
                        max_tokens_val = int(task.get("max_tokens") or self.n_predict)
                        per_seq_ctx = max(256, self.n_ctx // max(1, self.continuous_batch_slots))
                        # Reserve enough room for generation (at least min(max_tokens_val, 512) tokens)
                        reserve_gen = min(max_tokens_val, max(256, per_seq_ctx // 3))
                        max_prompt = max(128, per_seq_ctx - reserve_gen)
                        if len(prompt_tokens) > max_prompt:
                            # Keep system/header prefix + tail (user question & latest context)
                            head_keep = min(256, max_prompt // 4)
                            tail_keep = max_prompt - head_keep
                            prompt_tokens = prompt_tokens[:head_keep] + prompt_tokens[-tail_keep:]
                        active_model_name = list(self.current_model.keys())[0] if self.current_model else None
                        stops_val = self.get_stops_for_model(active_model_name, task.get("stop"))

                        active_seqs[slot_id] = {
                            "task": task,
                            "prompt_tokens": prompt_tokens,
                            "prompt_token_count": len(prompt_tokens),
                            "prefill_idx": 0,
                            "pos": 0,
                            "next_token": 0,
                            "generated_count": 0,
                            "max_tokens": int(max_tokens_val),
                            "stops": stops_val or [],
                            "byte_buf": bytearray(),
                            "text_buf": "",
                            "sampler": self._build_sampler(model, task),
                        }
                    except Exception as admit_exc:
                        if not future.done():
                            future.set_exception(admit_exc)
                        if task.get("stream") and task.get("stream_queue"):
                            await task["stream_queue"].put(admit_exc)
                            await task["stream_queue"].put(None)
                        self.waiting_queue.task_done()
                        await self._available_slots.put(slot_id)
                        await self._release_model()

                # 2. If nothing active, wait briefly for next request (hold in pending_task without re-putting)
                if not active_seqs:
                    try:
                        pending_task = await asyncio.wait_for(self.waiting_queue.get(), timeout=0.5)
                        continue
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

                # 3. Check cancellations before stepping
                for seq_id in list(active_seqs.keys()):
                    seq = active_seqs[seq_id]
                    task = seq["task"]
                    tid = task["task_id"]
                    cancel_flag = self._stream_cancel_flags.get(tid)
                    if task["future"].cancelled() or (cancel_flag and cancel_flag.is_set()):
                        await asyncio.to_thread(self._release_seq_sync, model, seq_id, seq["sampler"])
                        if task.get("stream") and task.get("stream_queue"):
                            await task["stream_queue"].put(None)
                        if not task["future"].done():
                            task["future"].set_result(None)
                        active_seqs.pop(seq_id)
                        self.waiting_queue.task_done()
                        await self._available_slots.put(seq_id)
                        await self._release_model()

                if not active_seqs:
                    continue

                # 4. Run ONE batched GPU/CPU decode step for all active sequences
                try:
                    events = await asyncio.to_thread(self._step_continuous_batch_sync, model, active_seqs)
                except Exception as step_exc:
                    logger.error(f"Continuous batch step failed: {step_exc}", exc_info=True)
                    for seq_id in list(active_seqs.keys()):
                        seq = active_seqs.pop(seq_id)
                        task = seq["task"]
                        await asyncio.to_thread(self._release_seq_sync, model, seq_id, seq["sampler"])
                        if not task["future"].done():
                            task["future"].set_exception(step_exc)
                        if task.get("stream") and task.get("stream_queue"):
                            await task["stream_queue"].put(step_exc)
                            await task["stream_queue"].put(None)
                        self.waiting_queue.task_done()
                        await self._available_slots.put(seq_id)
                        await self._release_model()
                    continue

                # 5. Dispatch streamed tokens & retire finished sequences immediately
                for seq_id, piece, finished, finish_reason in events:
                    seq = active_seqs.get(seq_id)
                    if not seq:
                        continue
                    task = seq["task"]
                    if task.get("stream") and task.get("stream_queue") and piece:
                        await task["stream_queue"].put(piece)

                    if finished:
                        await asyncio.to_thread(self._release_seq_sync, model, seq_id, seq["sampler"])
                        active_seqs.pop(seq_id, None)
                        prompt_cnt = seq["prompt_token_count"]
                        comp_cnt = seq["generated_count"]
                        exact_usage = {
                            "prompt_tokens": prompt_cnt,
                            "completion_tokens": comp_cnt,
                            "total_tokens": prompt_cnt + comp_cnt,
                        }
                        self._last_usage[task["task_id"]] = exact_usage
                        if task.get("stream") and task.get("stream_queue"):
                            await task["stream_queue"].put(None)
                            if not task["future"].done():
                                task["future"].set_result(None)
                        else:
                            if not task["future"].done():
                                msg = AIMessage(
                                    content=seq["text_buf"].strip(),
                                    response_metadata={
                                        "finish_reason": finish_reason,
                                        "task_id": task["task_id"],
                                        "seq_id": seq_id,
                                        "generated_tokens": comp_cnt,
                                        "usage": exact_usage,
                                    },
                                )
                                task["future"].set_result(msg)
                        self.waiting_queue.task_done()
                        await self._available_slots.put(seq_id)
                        await self._release_model()
        finally:
            for seq_id, seq in list(active_seqs.items()):
                if model is not None:
                    await asyncio.to_thread(self._release_seq_sync, model, seq_id, seq["sampler"])
                task = seq["task"]
                if not task["future"].done():
                    task["future"].set_exception(asyncio.CancelledError("Worker stopped"))
                if task.get("stream") and task.get("stream_queue"):
                    await task["stream_queue"].put(None)
                self.waiting_queue.task_done()
                await self._available_slots.put(seq_id)
                await self._release_model()

    async def start(self):
        if self._running_event.is_set():
            logger.warning("Worker already running")
            return
        
        if not self.current_model:
            raise RuntimeError("No model loaded. Call _load_model() before start().")
        
        self._running_event.set()
        self._worker_task = asyncio.create_task(self._worker())
        self._worker_task.add_done_callback(self._worker_done_callback)
        logger.info(f"Service started with Continuous Batching ({self.continuous_batch_slots} slots)")

    async def stop(self):
        if not self._running_event.is_set():
            return
        
        logger.info("Stopping service...")
        
        # 1. Stop accepting new requests
        self._running_event.clear()
        
        # 2. Wait for active requests to finish
        async with self._model_lock:
            while self._active_requests > 0:
                await asyncio.sleep(0.1)
        
        # 3. Cancel streams
        for flag in self._stream_cancel_flags.values():
            flag.set()
        self._stream_cancel_flags.clear()
        
        # 4. Cancel active stream tasks & continuous batch slot tasks
        for task_id, task in list(self._stream_tasks.items()):
            task.cancel()
            logger.info(f"Cancelled stream {task_id}")
        if self._stream_tasks:
            await asyncio.gather(*self._stream_tasks.values(), return_exceptions=True)
        self._stream_tasks.clear()

        for task_id, task in list(self._active_slot_tasks.items()):
            task.cancel()
            logger.info(f"Cancelled continuous batch slot task {task_id}")
        if self._active_slot_tasks:
            await asyncio.gather(*self._active_slot_tasks.values(), return_exceptions=True)
        self._active_slot_tasks.clear()
        
        # 5. Stop the scheduler worker
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._worker_task = None
        
        # 6. Fail pending futures in waiting queue
        failed_count = 0
        while not self.waiting_queue.empty():
            try:
                task = self.waiting_queue.get_nowait()
                future = task.get("future")
                if future and not future.done():
                    future.set_exception(RuntimeError("Service stopped"))
                    failed_count += 1
                self.waiting_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # 7. Fail any futures still in self.futures
        async with self._future_lock:
            for fid, fut in self.futures.items():
                if not fut.done():
                    fut.set_exception(RuntimeError("Service stopped"))
            self.futures.clear()
        
        logger.info(f"Service stopped. Failed {failed_count} pending requests.")

    # ------------------------ Core APIs ------------------------
    def _extract_content(self, response: Any) -> str:
        """Helper to extract text content from raw response dict"""
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    return choice["text"].strip()
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"].strip()
        return str(response)

    async def chat_llm(self, chat: str, chat_id: Optional[str] = None, timeout: float = None, **gen_kwargs) -> AIMessage:
        if timeout is None:
            timeout = self.config.default_timeout
        
        async with self.lock:
            if not self.current_model:
                raise ValueError("No model loaded. Call _load_model() first.")
        
        if not self._running_event.is_set():
            raise RuntimeError("Service not started")
        
        task_id = chat_id or str(uuid4())[:12]
        future = asyncio.Future()
        
        async with self._future_lock:
            if task_id in self.futures:
                raise ValueError(f"Request ID '{task_id}' already in use")
            self.futures[task_id] = future
        
        async with self._metrics_lock:
            self._metrics["total_requests"] += 1
        
        task_params = {
            "task_id": task_id,
            "prompt": chat,
            "future": future,
            "timestamp": time.time(),
            "stream": False,
        }
        allowed_params = ["temperature", "top_p", "top_k", "max_tokens", "stop", "repeat_penalty", "response_format", "model_id", "user_id"]
        for p in allowed_params:
            if p in gen_kwargs and gen_kwargs[p] is not None:
                task_params[p] = gen_kwargs[p]
        
        try:
            if self.config.reject_on_full_queue:
                self.queue.put_nowait(task_params)
            else:
                await self.queue.put(task_params)
        except asyncio.QueueFull:
            async with self._future_lock:
                self.futures.pop(task_id, None)
            raise asyncio.QueueFull(f"Queue is full (max {self.config.queue_maxsize})")
        
        logger.info(f"Request {task_id} queued (size: {self.queue.qsize()})")
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            async with self._metrics_lock:
                self._metrics["completed_requests"] += 1
            return result
        except Exception as e:
            async with self._metrics_lock:
                self._metrics["total_errors"] += 1
            logger.error(f"Request {task_id} failed: {e}")
            raise
        finally:
            # ✅ REMOVED: chat_llm should NOT release model (worker does that)
            async with self._future_lock:
                self.futures.pop(task_id, None)

    async def stream_llm(self, chat: str, chat_id: Optional[str] = None, timeout: float = None, **gen_kwargs):
        if timeout is None:
            timeout = self.config.default_timeout
        
        async with self.lock:
            if not self.current_model:
                raise ValueError("No model loaded")
        if not self._running_event.is_set():
            raise RuntimeError("Service not started")

        task_id = chat_id or str(uuid4())[:12]
        cancel_flag = asyncio.Event()
        self._stream_cancel_flags[task_id] = cancel_flag
        
        async with self._stream_lock:
            if task_id in self._active_streams:
                raise ValueError(f"Stream ID '{task_id}' already active")
            self._active_streams.add(task_id)
        
        async with self._metrics_lock:
            self._metrics["streaming_requests"] += 1
            self._metrics["total_requests"] += 1

        stream_queue = asyncio.Queue()
        future = asyncio.Future()

        task_params = {
            "task_id": task_id,
            "prompt": chat,
            "future": future,
            "timestamp": time.time(),
            "stream": True,
            "stream_queue": stream_queue,
        }
        allowed_params = ["temperature", "top_p", "top_k", "max_tokens", "stop", "repeat_penalty", "response_format", "model_id", "user_id"]
        for p in allowed_params:
            if p in gen_kwargs and gen_kwargs[p] is not None:
                task_params[p] = gen_kwargs[p]

        await self.queue.put(task_params)

        async def monitor():
            try:
                await future
            except Exception as e:
                await stream_queue.put(e)
                await stream_queue.put(None)
            finally:
                async with self._stream_lock:
                    self._active_streams.discard(task_id)

        mt = asyncio.create_task(monitor())

        try:
            while True:
                if cancel_flag.is_set():
                    raise asyncio.CancelledError(f"Stream {task_id} cancelled")
                try:
                    token = await asyncio.wait_for(stream_queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    if not future.done():
                        future.set_exception(asyncio.TimeoutError(f"Stream timed out after {timeout}s"))
                    break
                
                if token is None:
                    break
                if isinstance(token, Exception):
                    raise token
                yield token
        finally:
            self._stream_cancel_flags.pop(task_id, None)
            mt.cancel()
            try:
                await mt
            except asyncio.CancelledError:
                pass
            async with self._stream_lock:
                self._active_streams.discard(task_id)
    # ------------------------ Runnable interface ------------------------
    async def ainvoke(self, input: Union[str, Dict]) -> AIMessage:
        if isinstance(input, dict):
            prompt = input.get("input", str(input))
            chat_id = input.get("chat_id")
            timeout = input.get("timeout", self.config.default_timeout)
            gen_kwargs = {k: v for k, v in input.items() if k in ["temperature", "top_p", "top_k", "max_tokens", "stop", "repeat_penalty","user_id", "model_id"]}
        else:
            prompt = str(input)
            chat_id = None
            timeout = self.config.default_timeout
            gen_kwargs = {}
        return await self.chat_llm(prompt, chat_id=chat_id, timeout=timeout, **gen_kwargs)

    def invoke(self, input: Union[str, Dict]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input))
        else:
            return loop.run_until_complete(self.ainvoke(input))

    def __or__(self, other):             
        async def chained(value):
            response = await self.ainvoke(value)
            if hasattr(other, "ainvoke"):
                return await other.ainvoke(response)
            elif hasattr(other, "invoke"):
                return other.invoke(response)
            elif callable(other):
                return await other(response) if asyncio.iscoroutinefunction(other) else other(response)
            return response
        return Runnable(chained)

    def __ror__(self, other):
        if callable(other):
            async def chained(value):
                processed = other(value)
                return await self.ainvoke(processed)
            return chained
        return self

    async def __call__(self, input: Union[str, Dict]) -> str:
        return await self.ainvoke(input)

    # ------------------------ Unload and memory ------------------------
    async def get_loaded_models(self) -> List[str]:
        async with self.lock:
            return list(self.current_model.keys())

    async def get_memory_usage(self):
        async with self.lock:
            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            return {
                "rss_mb": mem.rss / (1024 * 1024),
                "vms_mb": mem.vms / (1024 * 1024),
                "models_loaded": len(self.current_model),
                "model_names": list(self.current_model.keys())
            }

    async def unload_model(self, model_name: str = None):
        async with self._model_lock:
            self._unload_allowed.clear()
            while self._active_requests > 0:
                await asyncio.sleep(0.1)
        
        async with self.lock:
            if model_name is None:
                for name in list(self.current_model.keys()):
                    await self._unload_single_model_async(name)
                self.current_model.clear()
                logger.info("Unloaded all models")
                return True
            if model_name not in self.current_model:
                raise ValueError(f"Model '{model_name}' not loaded")
            return await self._unload_single_model_async(model_name)

    async def _unload_single_model_async(self, model_name: str) -> bool:
        try:
            logger.info(f"Unloading model: {model_name}")
            model = self.current_model.pop(model_name, None)
            slots = self._model_slots.pop(model_name, [])
            for s in slots:
                if hasattr(s, "close"):
                    try:
                        s.close()
                    except Exception:
                        pass
            if model and hasattr(model, "close") and model not in slots:
                try:
                    model.close()
                except Exception:
                    pass
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._gc_cleanup)
            logger.info(f"Model {model_name} unloaded")
            return True
        except Exception as e:
            logger.error(f"Error unloading {model_name}: {e}")
            return False
    def _gc_cleanup(self):
        gc.collect()
        import sys
        if sys.platform.startswith('linux'):
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass

    async def _acquire_slot_model(self, slot_id: int):
        """Wait until unload is not in progress, then increment counter and return assigned slot model."""
        await self._unload_allowed.wait()
        async with self._model_lock:
            self._active_requests += 1
            async with self.lock:
                if not self.current_model:
                    raise ValueError("No model loaded")
                model_name = list(self.current_model.keys())[0]
                slots = self._model_slots.get(model_name)
                if slots and slot_id < len(slots):
                    return slots[slot_id]
                return list(self.current_model.values())[0]

    async def _acquire_model(self):
        """Wait until unload is not in progress, then increment counter and return model."""
        await self._unload_allowed.wait()
        async with self._model_lock:
            self._active_requests += 1
            async with self.lock:
                if not self.current_model:
                    raise ValueError("No model loaded")
                return list(self.current_model.values())[0]

    async def _release_model(self):
        async with self._model_lock:
            self._active_requests -= 1
            if self._active_requests == 0:
                self._unload_allowed.set()

    async def unload_all_models(self):
        return await self.unload_model(None)