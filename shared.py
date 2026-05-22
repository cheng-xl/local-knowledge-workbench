from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def human_message(content: str) -> dict:
    return {"role": "user", "content": content}


def ai_message(content: str) -> dict:
    return {"role": "assistant", "content": content}


def tool_message(content: str, tool_call_id: str = "", name: str = "") -> dict:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id, "name": name}
