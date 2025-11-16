from pydantic import BaseModel
from typing import List, Dict, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    # 시스템 프롬프트는 클라이언트가 관리하거나 여기서 고정할 수 있음
    # 여기서는 클라이언트가 전체 기록을 보낸다고 가정
    chat_history: List[ChatMessage]

class ChatResponse(BaseModel):
    ai_response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None 