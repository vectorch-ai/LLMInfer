__version__ = "0.1.3"
try:
    # torch needs to be imported first, otherwise it will segfault upon import.
    import torch  # noqa: F401
except ImportError:
    pass

from scalellm._C import (LLMHandler, LogProb, LogProbData, Message, Priority,
                         RequestOutput, SamplingParams, SequenceOutput, Status,
                         StatusCode, Usage, get_metrics)
from scalellm.errors import ValidationError
from scalellm.llm import LLM
from scalellm.llm_engine import AsyncLLMEngine, OutputAsyncStream, OutputStream

__all__ = [
    "Message",
    "LLM",
    "LogProb",
    "LogProbData",
    "AsyncLLMEngine",
    "OutputAsyncStream",
    "OutputStream",
    "ValidationError",
    "Priority",
    "RequestOutput",
    "SamplingParams",
    "SequenceOutput",
    "Status",
    "StatusCode",
    "Usage",
    "LLMHandler",
    "get_metrics",
]
