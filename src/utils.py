"""
Utility functions for the chat application.
"""
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

from models import QAPair, ResponseType
from type_definitions import ChatAgentProtocol

# We avoid the circular import by using TYPE_CHECKING for type hints only
# This import is never executed at runtime
if TYPE_CHECKING:
    from agent import ChatAgent

async def update_qa_dictionary(
    agent: ChatAgentProtocol,
    response: str,
    qa_dict: List[Dict[str, QAPair]],
    user_input: str
) -> bool:
    """
    Update the QA dictionary based on the response.
    
    Args:
        agent: Chat agent instance
        response: Response to analyze - this is the agent's response
        qa_dict: QA dictionary to update
        user_input: User's input that triggered the response
        
    Returns:
        bool: True if the dictionary was updated, False otherwise
    """
    # Get a logger for this function
    logger = logging.getLogger("chat_app")
    
    was_updated = False
    logger.info("Analyzing response for QA dictionary updates...")
    
    # For extraction, combine the user input and agent response
    combined_text = f"User: {user_input}\nAgent: {response}"
    
    # Determine what question was asked in the previous agent response
    # We'll check the conversation history to see what question was asked
    asked_fields = []
    
    # Get the lowercase agent response for easier matching
    response_lower = response.lower()
    
    # Look for questions about specific fields in the agent's response
    question_indicators = {
        "age": ["age", "how old", "years old", "your age", "current age", "birth year"],
        "location": ["where", "location", "country", "city", "live", "from", "located", "reside", "based"],
        "occupation": ["occupation", "job", "work", "profession", "career", "do for", "your job", "employment"],
        "education": ["education", "degree", "school", "college", "university", "studied", "graduate", "academic"]
    }
    
    # Track which fields were directly asked vs inferred
    explicitly_asked = []
    for field, indicators in question_indicators.items():
        if any(indicator in response_lower for indicator in indicators):
            explicitly_asked.append(field)
            asked_fields.append(field)
            logger.info(f"Detected question about '{field}' in agent response")
    
    # If we couldn't detect any questions, try to infer from missing information
    if not asked_fields:
        for entry in qa_dict:
            for key, value in entry.items():
                # Look for fields that are missing information and have few attempts
                if value.get("response") is None and value.get("attempts", 0) < 3:
                    asked_fields.append(key)
                    logger.info(f"Inferring question about '{key}' from missing information")
                    break  # Only infer one field at a time
    
    # Helper function to clean JSON/markdown formatting
    def clean_response_text(text: str) -> str:
        if not isinstance(text, str):
            return text
            
        # Remove markdown code blocks
        if "```" in text:
            text = text.replace("```json", "").replace("```", "")
            
        # Remove JSON characters
        for char in ['{', '}', '[', ']', '"', "'", '`']:
            text = text.replace(char, '')
            
        return text.strip()
    
    # Only process each entry in the QA dictionary for the fields that were asked about
    for entry in qa_dict:
        for question_key, qa_pair in entry.items():
            # Skip questions we haven't asked about
            if question_key not in asked_fields:
                continue
                
            # Skip if we already have a valid response that isn't "refused"
            if qa_pair.get("response") is not None and qa_pair.get("response") != "refused":
                logger.info(f"Skipping '{question_key}' - already have response: {qa_pair['response']}")
                continue
            
            # Classify the response
            classification = await agent.classify_response(combined_text, question_key)
            logger.info(f"Classification for '{question_key}': {classification}")
            
            # Only increment attempts for questions that were actually asked
            was_updated = True
            is_explicitly_asked = question_key in explicitly_asked
            
            # Handle based on classification
            if classification == ResponseType.GIVEN:
                # We have an answer, extract it
                extracted_info = await agent.extract_information(combined_text, question_key)
                
                # Special case for coding-related occupation
                if question_key == "occupation" and not extracted_info:
                    # More specific mapping of activities to professional titles
                    occupation_keywords = {
                        "code": "software developer",
                        "coding": "software developer",
                        "program": "software engineer",
                        "programming": "software engineer",
                        "develop": "software developer",
                        "developer": "software developer",
                        "tech": "technology professional",
                        "teach": "teacher",
                        "design": "designer",
                        "write": "writer",
                        "manage": "manager"
                    }
                    
                    # Find the first matching keyword
                    for keyword, title in occupation_keywords.items():
                        if keyword in user_input.lower():
                            extracted_info = title
                            logger.info(f"Occupation keyword match: '{keyword}' → '{title}'")
                            break
                
                # Special case for education to combine previous partial responses
                if question_key == "education":
                    # Check if we have a previous partial response to combine
                    prev_response = qa_pair.get("response")
                    if prev_response and extracted_info:
                        combined = f"{prev_response} {extracted_info}"
                        # Normalize combined result
                        combined = combined.replace("univer sity", "university").strip()
                        extracted_info = combined
                        logger.info(f"Combined education response: '{prev_response}' + '{extracted_info}' → '{combined}'")
                
                if extracted_info is not None:
                    # Clean and validate the extracted information
                    extracted_info = clean_response_text(extracted_info)
                    
                    # Field-specific validation
                    if question_key == "age" and not extracted_info.isdigit():
                        import re
                        # Try to extract just digits for age
                        age_match = re.search(r'\b(\d+)\b', extracted_info)
                        if age_match:
                            extracted_info = age_match.group(1)
                        else:
                            logger.warning(f"Invalid age format: {extracted_info}")
                            extracted_info = None
                    
                    # Truncate overly long values
                    if extracted_info and len(extracted_info) > 100:
                        logger.warning(f"Truncating overlength value for '{question_key}': {extracted_info}")
                        extracted_info = extracted_info[:100]
                    
                    # Update the QA pair with valid data
                    if extracted_info:
                        qa_pair["response"] = extracted_info
                        # Only increment attempts if explicitly asked
                        if is_explicitly_asked:
                            qa_pair["attempts"] = qa_pair.get("attempts", 0) + 1
                        logger.info(f"Updated '{question_key}' with validated response: {extracted_info}")
                    else:
                        # Increment attempt counter only if we failed to extract valid info and explicitly asked
                        if is_explicitly_asked:
                            qa_pair["attempts"] = qa_pair.get("attempts", 0) + 1
                            logger.info(f"Failed to extract valid info for '{question_key}', incrementing attempts to {qa_pair['attempts']}")
                else:
                    # Increment attempt counter if extraction failed and explicitly asked
                    if is_explicitly_asked:
                        qa_pair["attempts"] = qa_pair.get("attempts", 0) + 1
                        logger.info(f"Failed to extract info for '{question_key}', incrementing attempts to {qa_pair['attempts']}")
                    
            elif classification == ResponseType.REJECTED:
                # User explicitly refused, mark as "refused" and max out attempts
                qa_pair["response"] = "refused"
                qa_pair["attempts"] = 3  # No need to ask again
                logger.info(f"User refused to provide '{question_key}', marked as refused")
                
            else:
                # For UNSURE or INCOMPLETE, increment the attempt counter only if explicitly asked
                if is_explicitly_asked:
                    qa_pair["attempts"] = qa_pair.get("attempts", 0) + 1
                    logger.info(f"Unclear response for '{question_key}', incrementing attempts to {qa_pair['attempts']}")
    
    return was_updated

def format_response(response: str) -> str:
    """
    Format the response for display.
    
    Args:
        response: Raw response from the LLM
        
    Returns:
        str: Formatted response
    """
    # Simple formatter that removes extra whitespace
    return response.strip()

def setup_logger() -> logging.Logger:
    """
    Set up a logger for the application that writes to a file.
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger("chat_app")
    
    # Clear any existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()
        
    logger.setLevel(logging.DEBUG)
    
    # Create file handler instead of console handler
    log_file = os.path.join(os.path.dirname(__file__), "chat_app.log")
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger