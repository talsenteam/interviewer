"""
Fast agent implementation for the chat application.
"""
import json
import asyncio
from typing import Dict, List, Any, AsyncGenerator, Optional

import litellm
from tools import classify_response, extract_information
from models import QAPair, ResponseType
from utils import setup_logger

logger = setup_logger()

class ChatAgent:
    """
    Fast chat agent implementation that uses litellm for API calls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chat agent with the given configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_config = config["llm"]
        
        # Configure litellm
        litellm.api_key = self.llm_config["api_key"]
        if self.llm_config.get("api_base"):
            litellm.api_base = self.llm_config["api_base"]
    
    def _create_prompt(self, user_input: str, conversation_history: List[Dict[str, str]], qa_dict: List[Dict[str, QAPair]]) -> List[Dict[str, str]]:
        """
        Create the prompt for the LLM including conversation history and QA dictionary.
        
        Args:
            user_input: User's input
            conversation_history: Previous conversation history
            qa_dict: QA dictionary
            
        Returns:
            List[Dict[str, str]]: Messages for the LLM
        """
        # Convert QA dictionary to a readable format for the prompt
        qa_dict_str = json.dumps(qa_dict, indent=2)
        
        # Calculate missing fields to explicitly tell the agent what to ask about
        missing_fields = []
        fields_with_attempts = []
        collected_info = []
        
        for entry in qa_dict:
            for key, value in entry.items():
                # Track fields with missing responses
                if value.get("response") is None:
                    # Check if we've already asked 3 times
                    if value.get("attempts", 0) >= 3:
                        fields_with_attempts.append(f"{key} (max attempts reached)")
                    else:
                        missing_fields.append(key)
                elif value.get("response") == "refused":
                    # User explicitly refused
                    collected_info.append(f"{key}: refused to answer")
                else:
                    # Information collected
                    collected_info.append(f"{key}: {value.get('response')}")
        
        # Find the next question to ask based on sequence
        next_question = None
        question_sequence = ["location", "age", "occupation", "education"]
        for field in question_sequence:
            if field in missing_fields:
                next_question = field
                break
        
        system_prompt = f"""You are an interviewer collecting profile information from the user.

CURRENT STATUS:
- Information collected: {', '.join(collected_info) if collected_info else 'None yet'}
- Missing information: {', '.join(missing_fields) if missing_fields else 'All information collected!'}
- Fields to skip (max attempts reached): {', '.join(fields_with_attempts) if fields_with_attempts else 'None'}

YOUR IMMEDIATE GOAL: {f"Ask about the user's {next_question}" if next_question else "Thank the user, all information collected or attempted"}

CONVERSATION RULES:
1. Ask about ONLY ONE missing field per response - focus on "{next_question}" next
2. Ask direct questions: "What is your occupation?" rather than "Could you tell me about your occupation?"
3. Acknowledge the user's previous answer before asking a new question
4. If a user refuses to answer, clearly mark it by saying "I understand you prefer not to share your {next_question}."
5. Never use technical language or formatting like JSON
6. Keep responses concise and focused

QUESTION FORMAT EXAMPLES:
- Age: "How old are you?" or "What is your age?"
- Location: "Where are you from?" or "Where do you live?"
- Occupation: "What do you do for work?" or "What is your occupation?"
- Education: "What is your highest level of education?"

If all information is collected or max attempts reached, thank the user for their time.
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add latest user message
        if not conversation_history or conversation_history[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_input})
            
        return messages
    
    async def process_input(self, user_input: str, conversation_history: List[Dict[str, str]], qa_dict: List[Dict[str, QAPair]]) -> AsyncGenerator[str, None]:
        """
        Process user input and generate a response.
        
        Args:
            user_input: User's input
            conversation_history: Previous conversation history
            qa_dict: QA dictionary
            
        Yields:
            str: Response chunks
        """
        messages = self._create_prompt(user_input, conversation_history, qa_dict)
        
        try:
            # Stream the response from the LLM
            response = await litellm.acompletion(
                model=self.llm_config["model"],
                messages=messages,
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
                stream=True
            )
            
            async for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            yield f"\nI'm sorry, I encountered an error: {str(e)}"
    
    async def classify_response(self, response: str, question_key: str) -> ResponseType:
        """
        Classify a response using the classification tool.
        
        Args:
            response: Response to classify
            question_key: Key of the question being classified
            
        Returns:
            ResponseType: Classification result
        """
        return await classify_response(
            response=response,
            question_key=question_key,
            config=self.config["tools"]["classification"]
        )
    
    async def extract_information(self, response: str, question_key: str) -> Optional[str]:
        """
        Extract information from a response using the extraction tool.
        
        Args:
            response: Response to extract information from
            question_key: Key of the question to extract
            
        Returns:
            Optional[str]: Extracted information or None if extraction failed
        """
        return await extract_information(
            response=response,
            question_key=question_key,
            config=self.config["tools"]["extraction"]
        )