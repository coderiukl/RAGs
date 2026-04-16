from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str

class ChatHistory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._message: list[Message] = []
    
    def add(self, role: Literal["user", "assistant"], content: str):
        self._message.append(Message(role=role, content=content))

        if len(self._message) > self.max_turns * 2:
            self._message = self._message[-(self.max_turns * 2):]
        
    def get_message(self) -> list[Message]:
        return self._message
    
    def format_for_prompt(self) -> str:
        if not self._message:
            return ""
        
        lines = []
        for message in self._message:
            prefix = "User" if message.role == "user" else "Assistant"
            lines.append(f"{prefix}: {message.content}")
        return "\n".join(lines)
    
    def clear(self):
        self._message = []

    def __len__(self):
        return len(self._message)
