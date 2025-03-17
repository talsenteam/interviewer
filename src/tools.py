"""
Classification and extraction tools for the chat application.
"""
import json
from typing import Dict, Any, Optional

import litellm
from models import ResponseType, ExtractedInfo
from utils import setup_logger

logger = setup_logger()

async def classify_response(response: str, question_key: str, config: Dict[str, Any]) -> ResponseType:
    """
    Classify a response as given, rejected, unsure, or incomplete.
    
    Args:
        response: Response to classify
        question_key: Key of the question being classified
        config: Configuration for the classification tool
        
    Returns:
        ResponseType: Classification result
    """
    # Create a focused prompt to analyze the conversation
    classification_prompt = f"""
TASK: Determine if the user has provided information about their "{question_key}" in this conversation.

CONVERSATION:
"{response}"

CLASSIFY AS ONE OF THESE OPTIONS:
1. "given" - User has clearly provided information about their {question_key}
2. "rejected" - User has explicitly refused to provide information about their {question_key}
3. "unsure" - Cannot determine if user has provided information about their {question_key}
4. "incomplete" - User has partially provided information about their {question_key}

EXAMPLES OF "given" RESPONSES:
- For age: "I'm 30", "thirty", "in my 30s"
- For location: "I live in Paris", "from France", "New York"
- For occupation: "software engineer", "I work in healthcare", "teacher", "coder", "programmer", "developer", "IT professional", "tech worker", "freelancer", "consultant", "I code", "I program", "develop" 
- For education: "college degree", "high school", "I have a PhD"

EXAMPLES OF "rejected" RESPONSES:
- "I don't want to tell you my {question_key}"
- "That's private information"
- "I'd rather not say"
- "None of your business"
- "I'm not comfortable sharing that"
- "No" (when asked specifically about {question_key})
- "I prefer to keep that private"
- "I don't see why that's relevant"

EXAMPLES OF "unsure" RESPONSES:
- User changes the subject without answering
- User asks a counter-question without providing information
- No mention of the specific field at all

EXAMPLES OF "incomplete" RESPONSES:
- Partial information that needs clarification
- Vague references that aren't specific enough
- Only mention of a related field without clear detail
- For education: "univer" for "university"

YOUR RESPONSE MUST BE EXACTLY ONE WORD: "given", "rejected", "unsure", or "incomplete"
"""
    
    try:
        classification_response = await litellm.acompletion(
            model=config["model"],
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=config["temperature"],
            max_tokens=10  # Just need a single word
        )
        
        result = classification_response.choices[0].message.content.strip().lower()
        logger.info(f"Classification for '{question_key}': {result}")
        
        # Log both raw result and parsed enum
        if "given" in result:
            logger.info(f"Classification for '{question_key}': ResponseType.GIVEN")
            return ResponseType.GIVEN
        elif any(word in result for word in ["reject", "refused", "decline"]):
            logger.info(f"Classification for '{question_key}': ResponseType.REJECTED")
            return ResponseType.REJECTED
        elif "unsure" in result:
            logger.info(f"Classification for '{question_key}': ResponseType.UNSURE")
            return ResponseType.UNSURE
        elif "incomplete" in result:
            logger.info(f"Classification for '{question_key}': ResponseType.INCOMPLETE")
            return ResponseType.INCOMPLETE
        # Add direct keyword pattern matching for occupation as a fallback
        elif question_key == "occupation" and any(job in response.lower() for job in ["coder", "programmer", "developer", "engineer", "work as", "working as", "code", "coding", "program"]):
            logger.info(f"Classification for '{question_key}': ResponseType.GIVEN (keyword match)")
            return ResponseType.GIVEN
        else:
            logger.warning(f"Unexpected classification result: {result}, defaulting to UNSURE")
            logger.info(f"Classification for '{question_key}': ResponseType.UNSURE (default)")
            return ResponseType.UNSURE
            
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return ResponseType.UNSURE

async def extract_information(response: str, question_key: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Extract information from a response.
    
    Args:
        response: Response to extract information from
        question_key: Key of the question to extract
        config: Configuration for the extraction tool
        
    Returns:
        Optional[str]: Extracted information or None if extraction failed
    """
    # Create field-specific examples with clear response formats
    examples = {
        "age": "For age: User says \"I'm 30 years old\" → extract \"30\"",
        "location": "For location: User says \"I live in Bulgaria\" → extract \"Bulgaria\"",
        "occupation": "For occupation: User says \"I work as a software engineer\" → extract \"software engineer\"",
        "education": "For education: User says \"I have a Bachelor's degree\" → extract \"Bachelor's degree\""
    }
    
    # Enhanced examples for occupation to improve extraction that are more general
    if question_key == "occupation":
        examples[question_key] = """For occupation:
- User says "I work as a software engineer" → extract "software engineer"
- User says "I'm a teacher" → extract "teacher"
- User says "coding" → extract "developer" 
- User says "I code" → extract "software developer"
- User says "I do programming" → extract "software engineer"
- User says "I'm in IT" → extract "IT specialist"
- User says "I work in healthcare" → extract "healthcare worker"
- User says "freelancing" → extract "freelancer"
- User says "I'm a student" → extract "student"
"""
    
    # Enhanced examples for education
    if question_key == "education":
        examples[question_key] = """For education:
- User says "I have a Bachelor's degree" → extract "Bachelor's degree"
- User says "university" → extract "university"
- User says "univer" → extract "university"
- User says "college" → extract "college degree"
- User says "I studied at university" → extract "university"
- User says "I have a degree" → extract "degree"
"""
    
    field_example = examples.get(question_key, f"For {question_key}: User provides value → extract just the value")
    
    # Field-specific extraction instructions
    field_instructions = {
        "age": "Extract ONLY the number (e.g., '30' not 'thirty' or '30 years old')",
        "location": "Extract ONLY the place name (e.g., 'Bulgaria' not 'I'm from Bulgaria')",
        "occupation": "Extract common job titles from activities. Map: 'code'→'software developer', 'program'→'software engineer', 'IT'→'IT specialist'. Return most specific reasonable title.",
        "education": "Extract education level or degree. Normalize partial inputs like 'univer' to 'university'. Be very flexible with education terms."
    }
    
    field_instruction = field_instructions.get(question_key, "Extract ONLY the relevant value")
    
    extraction_prompt = f"""
IMPORTANT: Your task is to extract EXACTLY the specific information about "{question_key}" from this conversation.

RULES:
1. {field_instruction}
2. If information is refused, return ONLY the word "refused" (no quotes, no JSON)
3. If no information is found, return ONLY the word "None" (no quotes, no JSON)
4. NEVER include JSON formatting, markdown, or quotation marks in your response
5. Respond with THE RAW VALUE ONLY - no explanations or formatting
6. Do your best to extract even minimal information - don't return "None" if you can identify a profession

EXAMPLE:
{field_example}

CONVERSATION TO ANALYZE:
"{response}"

YOUR RESPONSE MUST BE THE EXTRACTED VALUE ONLY, NOTHING ELSE.
"""
    
    try:
        extraction_response = await litellm.acompletion(
            model=config["model"],
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=config["temperature"],
            max_tokens=50
        )
        
        result = extraction_response.choices[0].message.content.strip()
        logger.info(f"Raw extraction for '{question_key}': {result}")
        
        # Handle specific values
        if result.lower() == "none" or result.lower() == "null":
            return None
        elif result.lower() == "refused" or result.lower() == "reject" or result.lower() == "no":
            return "refused"
            
        # Clean up the result - remove any unexpected formatting
        result = result.replace('```', '').replace('json', '').replace('{', '').replace('}', '').replace('"', '').replace("'", "").strip()
        
        # Apply field-specific validation and normalization
        if question_key == "age":
            # For age, extract numbers only
            import re
            age_match = re.search(r'\b(\d+)\b', result)
            if age_match:
                result = age_match.group(1)
            elif not result.isdigit():
                logger.warning(f"Extracted age is not a valid number: {result}")
                return None
        
        # Normalize education terms
        elif question_key == "education" and result:
            # Fix common partial inputs
            result = result.lower()
            if "univer" in result and "sity" not in result:
                result = result.replace("univer", "university")
            if "uni " in result:
                result = result.replace("uni ", "university ")
            if "coll" in result and "ege" not in result:
                result = result.replace("coll", "college")
            
            # Cleanup and normalize
            result = result.strip()
            logger.info(f"Normalized education value: {result}")
        
        return result if result else None
            
    except Exception as e:
        logger.error(f"Error in extraction: {e}")
        return None