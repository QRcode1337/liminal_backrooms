from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

@dataclass
class Message:
    role: str
    content: str
    ai_name: Optional[str] = None
    model: Optional[str] = None
    hidden: bool = False
    generated_image_path: Optional[str] = None
    # For branch indicators or other metadata
    _type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "role": self.role,
            "content": self.content,
        }
        if self.ai_name:
            data["ai_name"] = self.ai_name
        if self.model:
            data["model"] = self.model
        if self.hidden:
            data["hidden"] = self.hidden
        if self.generated_image_path:
            data["generated_image_path"] = self.generated_image_path
        if self._type:
            data["_type"] = self._type
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            ai_name=data.get("ai_name"),
            model=data.get("model"),
            hidden=data.get("hidden", False),
            generated_image_path=data.get("generated_image_path"),
            _type=data.get("_type")
        )

@dataclass
class Branch:
    id: str
    type: Literal["branch", "rabbithole", "fork", "main"]
    selected_text: str
    conversation: List[Message]
    parent: Optional[str] = None
    created_at: Optional[str] = None
    turn_count: int = 0
    history: List[Message] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "selected_text": self.selected_text,
            "conversation": [msg.to_dict() for msg in self.conversation],
            "parent": self.parent,
            "created_at": self.created_at,
            "turn_count": self.turn_count,
            "history": [msg.to_dict() for msg in self.history]
        }
