from typing import Literal, TypedDict, Optional, List


class ChatResponse(TypedDict):
    role: Literal["assistant"] | Literal["tool"] | Literal["status"]
    content: str
    tool_name: Optional[str]
    tool_status: Optional[Literal["calling", "processing", "completed", "error"]]
    tool_chain: Optional[List[str]]  # Chain of tool names being executed
