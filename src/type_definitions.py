"""
Type definitions to prevent circular imports.
"""
from typing import Dict, List, Any, Optional, Protocol, AsyncGenerator, TypedDict

from models import ResponseType

class ChatAgentProtocol(Protocol):
    """Protocol defining the interface of ChatAgent for type checking."""
    
    async def classify_response(self, response: str, question_key: str) -> ResponseType:
        """Classify a response."""
        ...
        
    async def extract_information(self, response: str, question_key: str) -> Optional[str]:
        """Extract information from a response."""
        ...
        
    async def process_input(self, user_input: str, conversation_history: List[Dict[str, str]], qa_dict: List[Dict[str, 'QAPair']]) -> AsyncGenerator[str, None]:
        """Process user input and generate a response."""
        ...

class QAPairDict(TypedDict):
    """TypedDict for QA pair in the dictionary."""
    response: Optional[str]
    attempts: int 