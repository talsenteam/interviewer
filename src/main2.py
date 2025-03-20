#!/usr/bin/env python3
"""
Interview agent using smolagents framework with litellm and custom tools.
This agent conducts structured interviews to collect profile information.
"""
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, AsyncGenerator

import litellm
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

# Models and type definitions
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

class Message(BaseModel):
    """Chat message model."""
    role: str
    content: str

# Tool definitions
class ClassifyResponseTool:
    """Tool to classify user responses."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the classification tool with configuration."""
        self.config = config
    
    async def __call__(self, response: str, question_key: str) -> ResponseType:
        """
        Classify a response as given, rejected, unsure, or incomplete.
        
        Args:
            response: Response to classify
            question_key: Key of the question being classified
            
        Returns:
            ResponseType: Classification result
        """
        classification_prompt = f"""
TASK: Determine if the user has provided information about their "{question_key}" in this conversation.

CONVERSATION:
"{response}"

CLASSIFY AS ONE OF THESE OPTIONS:
1. "given" - User has clearly provided information about their {question_key}
2. "rejected" - User has explicitly refused to provide information about their {question_key}
3. "unsure" - Cannot determine if user has provided information about their {question_key}
4. "incomplete" - User has partially provided information about their {question_key}

YOUR RESPONSE MUST BE EXACTLY ONE WORD: "given", "rejected", "unsure", or "incomplete"
"""
        
        try:
            classification_response = await litellm.acompletion(
                model=self.config["model"],
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=self.config["temperature"],
                max_tokens=10
            )
            
            result = classification_response.choices[0].message.content.strip().lower()
            
            if "given" in result:
                return ResponseType.GIVEN
            elif any(word in result for word in ["reject", "refused", "decline"]):
                return ResponseType.REJECTED
            elif "unsure" in result:
                return ResponseType.UNSURE
            elif "incomplete" in result:
                return ResponseType.INCOMPLETE
            else:
                return ResponseType.UNSURE
                
        except Exception as e:
            print(f"Error in classification: {e}")
            return ResponseType.UNSURE

class ExtractInformationTool:
    """Tool to extract specific information from user responses."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the extraction tool with configuration."""
        self.config = config
    
    async def __call__(self, response: str, question_key: str) -> Optional[str]:
        """
        Extract information from a response.
        
        Args:
            response: Response to extract information from
            question_key: Key of the question to extract
            
        Returns:
            Optional[str]: Extracted information or None if extraction failed
        """
        field_examples = {
            "age": "User says \"I'm 30 years old\" → extract \"30\"",
            "location": "User says \"I live in Bulgaria\" → extract \"Bulgaria\"",
            "occupation": "User says \"I work as a software engineer\" → extract \"software engineer\"",
            "education": "User says \"I have a Bachelor's degree\" → extract \"Bachelor's degree\""
        }
        
        field_example = field_examples.get(question_key, f"User provides {question_key} → extract just the value")
        
        extraction_prompt = f"""
IMPORTANT: Extract EXACTLY the specific information about "{question_key}" from this conversation.

RULES:
1. If information is refused, return ONLY the word "refused"
2. If no information is found, return ONLY the word "None"
3. Respond with THE RAW VALUE ONLY - no explanations

EXAMPLE:
{field_example}

CONVERSATION TO ANALYZE:
"{response}"

YOUR RESPONSE MUST BE THE EXTRACTED VALUE ONLY.
"""
        
        try:
            extraction_response = await litellm.acompletion(
                model=self.config["model"],
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=self.config["temperature"],
                max_tokens=50
            )
            
            result = extraction_response.choices[0].message.content.strip()
            
            if result.lower() in ["none", "null"]:
                return None
            elif result.lower() in ["refused", "reject", "no"]:
                return "refused"
                
            return result
                
        except Exception as e:
            print(f"Error in extraction: {e}")
            return None

class InterviewAgent:
    """Agent that conducts structured interviews to collect profile information."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the interview agent with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_config = config["llm"]
        
        # Configure litellm
        litellm.api_key = self.llm_config["api_key"]
        if self.llm_config.get("api_base"):
            litellm.api_base = self.llm_config["api_base"]
            
        # Initialize tools
        self.classify_tool = ClassifyResponseTool(config["tools"]["classification"])
        self.extract_tool = ExtractInformationTool(config["tools"]["extraction"])
    
    def _create_prompt(self, user_input: str, conversation_history: List[Dict[str, str]], 
                       qa_dict: List[Dict[str, QAPair]]) -> List[Dict[str, str]]:
        """Create the prompt for the LLM."""
        # Extract information about collected and missing fields
        missing_fields = []
        fields_with_attempts = []
        collected_info = []
        
        for entry in qa_dict:
            for key, value in entry.items():
                if value.get("response") is None:
                    if value.get("attempts", 0) >= 3:
                        fields_with_attempts.append(f"{key} (max attempts reached)")
                    else:
                        missing_fields.append(key)
                elif value.get("response") == "refused":
                    collected_info.append(f"{key}: refused to answer")
                else:
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
5. Keep responses concise and focused
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        
        # Add latest user message
        if not conversation_history or conversation_history[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_input})
            
        return messages
    
    async def process_input(self, user_input: str, conversation_history: List[Dict[str, str]], 
                           qa_dict: List[Dict[str, QAPair]]) -> AsyncGenerator[str, None]:
        """Process user input and generate a response."""
        messages = self._create_prompt(user_input, conversation_history, qa_dict)
        
        try:
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
            print(f"Error in LLM call: {e}")
            yield f"\nI'm sorry, I encountered an error: {str(e)}"
    
    async def classify_response(self, response: str, question_key: str) -> ResponseType:
        """Classify a response using the classification tool."""
        return await self.classify_tool(response, question_key)
    
    async def extract_information(self, response: str, question_key: str) -> Optional[str]:
        """Extract information from a response using the extraction tool."""
        return await self.extract_tool(response, question_key)

async def update_qa_dictionary(
    agent: InterviewAgent,
    response: str,
    qa_dict: List[Dict[str, QAPair]],
    user_input: str
) -> bool:
    """Update the QA dictionary based on the response."""
    was_updated = False
    
    # Combine user input and agent response for analysis
    combined_text = f"User: {user_input}\nAgent: {response}"
    
    # Check what question fields are mentioned in the response
    response_lower = response.lower()
    
    # Check for keywords related to each field in the QA dictionary
    field_keywords = {
        "age": ["age", "old", "years", "young"],
        "location": ["location", "live", "from", "country", "city", "area", "where"],
        "occupation": ["occupation", "job", "work", "profession", "career", "do for a living"],
        "education": ["education", "degree", "college", "university", "school", "studied"]
    }
    
    # Identify which fields might have been answered in this response
    potential_fields = []
    
    # First check what the agent asked about
    for field, keywords in field_keywords.items():
        if any(keyword in response_lower for keyword in keywords):
            potential_fields.append(field)
    
    # If no fields found in agent's response, check all fields in user input
    if not potential_fields:
        for field in ["age", "location", "occupation", "education"]:
            potential_fields.append(field)
    
    # For each potential field, try to extract and classify the response
    for field in potential_fields:
        # Find the QA pair for this field
        for entry in qa_dict:
            if field in entry:
                # Only process if we haven't already collected this info or reached max attempts
                if entry[field].get("response") is None and entry[field].get("attempts", 0) < 3:
                    # Classify the response
                    classification = await agent.classify_response(user_input, field)
                    
                    # Increment the attempt counter
                    entry[field]["attempts"] = entry[field].get("attempts", 0) + 1
                    was_updated = True
                    
                    # Extract information if the classification is GIVEN or INCOMPLETE
                    if classification in [ResponseType.GIVEN, ResponseType.INCOMPLETE]:
                        extracted_info = await agent.extract_information(user_input, field)
                        if extracted_info:
                            entry[field]["response"] = extracted_info
                    # Mark as refused if the classification is REJECTED
                    elif classification == ResponseType.REJECTED:
                        entry[field]["response"] = "refused"
    
    return was_updated

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    # Try to load from docker/.env first
    docker_env_path = os.path.join("docker", ".env")
    
    if os.path.exists(docker_env_path):
        load_dotenv(docker_env_path)
    else:
        # Fallback to local .env file
        load_dotenv()
    
    config = {
        # LLM Configuration
        "llm": {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            "api_key": os.getenv("LLM_API_KEY"),
            "api_base": os.getenv("LLM_API_BASE", None),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 2000)),
            "temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)),
        },
        # Tool Configuration
        "tools": {
            "classification": {
                "model": os.getenv("CLASSIFICATION_MODEL", "gpt-3.5-turbo"),
                "temperature": float(os.getenv("CLASSIFICATION_TEMPERATURE", 0.2)),
            },
            "extraction": {
                "model": os.getenv("EXTRACTION_MODEL", "gpt-3.5-turbo"),
                "temperature": float(os.getenv("EXTRACTION_TEMPERATURE", 0.2)),
            }
        },
        # App Configuration
        "app": {
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
    }
    
    return config

async def main():
    """Main application entry point."""
    # Load configuration
    config = load_config()
    
    # Initialize QA dictionary or load existing one
    qa_dict_path = "qa_dictionary.json"
    qa_dict = []
    
    if os.path.exists(qa_dict_path):
        try:
            with open(qa_dict_path, 'r') as f:
                qa_dict = json.load(f)
                # Basic validation
                if not isinstance(qa_dict, list):
                    qa_dict = []
        except json.JSONDecodeError:
            print("Error parsing qa_dictionary.json, starting with empty dictionary")
            qa_dict = []
    
    # If qa_dict is empty, initialize it with default values
    if not qa_dict:
        qa_dict = [
            {"age": {"response": None, "attempts": 0}},
            {"location": {"response": None, "attempts": 0}},
            {"occupation": {"response": None, "attempts": 0}},
            {"education": {"response": None, "attempts": 0}}
        ]
    
    # Initialize interview agent
    agent = InterviewAgent(config)
    
    # Chat loop
    conversation_history = []
    print("Welcome to the Interview Agent. Type 'exit' to quit.")
    
    try:
        # Display initial greeting
        initial_greeting = "Hello! I'm conducting a brief interview to gather some basic profile information. To start with, where are you from or where do you live?"
        print(f"\nAgent: {initial_greeting}")
        conversation_history.append({"role": "assistant", "content": initial_greeting})
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            
            # Process user input with agent
            print("\nAgent: ", end="", flush=True)
            response_chunks = []
            
            async for chunk in agent.process_input(user_input, conversation_history, qa_dict):
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
            
            full_response = "".join(response_chunks)
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": full_response})
            
            # Check for farewell messages
            farewell_phrases = ["goodbye", "take care", "chat session ended", "farewell"]
            if any(phrase in full_response.lower() for phrase in farewell_phrases):
                print("\nChat session ended.")
                break
            
            # Update QA dictionary based on the response
            updated = await update_qa_dictionary(agent, full_response, qa_dict, user_input)
            if updated:
                # Save the updated dictionary
                with open(qa_dict_path, 'w') as f:
                    json.dump(qa_dict, f, indent=2)
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    
    # Save final state
    with open(qa_dict_path, 'w') as f:
        json.dump(qa_dict, f, indent=2)
    
    print("Chat session ended.")

if __name__ == "__main__":
    asyncio.run(main()) 