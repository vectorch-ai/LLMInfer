from enum import Enum
from typing import Callable, List, Optional

# Defined in scalellm/csrc/scalellm.cpp
class SamplingParams:
    def __init__(self) -> None: ...
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    temperature: float
    top_p: float
    top_k: int

class ChatMessage:
    def __init__(self) -> None: ...
    role: str
    content: str

class Statistics:
    def __init__(self) -> None: ...
    num_prompt_tokens: int
    num_generated_tokens: int
    num_total_tokens: int

class SequenceOutput:
    def __init__(self) -> None: ...
    index: int
    text: str
    finish_reason: Optional[str]

class RequestOutput:
    def __init__(self) -> None: ...
    status: Optional[Status]
    outputs: List[SequenceOutput]
    stats: Optional[Statistics]
    finished: bool

class StatusCode(Enum):
    OK: StatusCode = ...
    CANCELLED: StatusCode = ...
    UNKNOWN: StatusCode = ...
    INVALID_ARGUMENT: StatusCode = ...
    DEADLINE_EXCEEDED: StatusCode = ...
    RESOURCE_EXHAUSTED: StatusCode = ...
    UNAUTHENTICATED: StatusCode = ...
    UNAVAILABLE: StatusCode = ...
    UNIMPLEMENTED: StatusCode = ...

class Status:
    def __init__(self, code: StatusCode, message: str) -> None: ...
    @property
    def code(self) -> StatusCode: ...
    @property
    def message(self) -> str: ...
    @property
    def ok(self) -> bool: ...

class LLMHandler:
    def __init__(self, model_path: str, devices: str) -> None: ...
    def schedule(
        self,
        prompt: str,
        sp: SamplingParams,
        callback: Callable[[RequestOutput], bool],
    ) -> bool: ...
    def stop(self) -> None: ...

# Defined in scalellm/csrc/llm.h
class LLM:
    def __init__(
        self,
        model_path: str,
        sampling_parameter: SamplingParams,
        max_seq_len: int,
        devices: str,
    ) -> None: ...
    def generate(self, batched_prompt: List[str]) -> None: ...
