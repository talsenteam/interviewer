"""
Pydantic models for data validation.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

class ResponseType(str, Enum):
    """Possible response types from classification."""
    GIVEN = "given"
    REJECTED = "rejected"
    UNSURE = "unsure"
    INCOMPLETE = "incomplete"

class QAPair(BaseModel):
    """Model for QA pair in the dictionary."""
    response: Optional[str] = None
    attempts: int = 0

class ExtractedInfo(BaseModel):
    """Model for extracted information."""
    key: str = Field(..., description="The key/question that was answered")
    value: str = Field(..., description="The extracted answer")

class ClassificationResult(BaseModel):
    """Result of the classification tool."""
    classification: ResponseType = Field(..., description="Classification of the response")
    confidence: float = Field(..., description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Reasoning behind the classification")

class Message(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: str = Field(..., description="Content of the message")

class ConversationContext(BaseModel):
    """Model for the conversation context."""
    history: List[Message] = Field(default_factory=list, description="Conversation history")
    qa_dict: List[Dict[str, QAPair]] = Field(default_factory=list, description="QA dictionary")