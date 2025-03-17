#!/usr/bin/env python3
"""
Terminal chat application that streams responses using litellm.
"""
import asyncio
import json
import os
from typing import Dict, List, Any, Optional

from agent import ChatAgent
from config import load_config
from models import QAPair, ResponseType
from utils import update_qa_dictionary, format_response, setup_logger

logger = setup_logger()

async def main():
    """Main application entry point."""
    # Load configuration from docker/.env
    config = load_config()
    
    # Initialize QA dictionary or load existing one
    qa_dict_path = os.path.join(os.path.dirname(__file__), "qa_dictionary.json")
    qa_dict: List[Dict[str, QAPair]] = []
    
    if os.path.exists(qa_dict_path):
        try:
            with open(qa_dict_path, 'r') as f:
                qa_dict = json.load(f)
                # Validate the loaded dictionary
                for entry in qa_dict:
                    for key, value in entry.items():
                        # Ensure each entry has the required fields
                        if not isinstance(value, dict) or "response" not in value or "attempts" not in value:
                            logger.warning(f"Invalid entry in qa_dictionary.json for key {key}. Reinitializing.")
                            qa_dict = []
                            break
        except json.JSONDecodeError:
            logger.error("Error parsing qa_dictionary.json, starting with empty dictionary")
            qa_dict = []
    
    # If qa_dict is empty, initialize it with default values
    if not qa_dict:
        qa_dict = [
            {"age": {"response": None, "attempts": 0}},
            {"location": {"response": None, "attempts": 0}},
            {"occupation": {"response": None, "attempts": 0}},
            {"education": {"response": None, "attempts": 0}}
        ]
    
    # Initialize chat agent
    agent = ChatAgent(config)
    
    # Chat loop
    conversation_history = []
    print("Welcome to the Chat Agent. Type 'exit' to quit.")
    
    try:
        # Display initial greeting from the agent
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
            
            # Check for farewell messages in agent response to terminate conversation
            farewell_phrases = ["goodbye", "take care", "chat session ended", "farewell"]
            if any(phrase in full_response.lower() for phrase in farewell_phrases):
                # Wait a moment before ending to let user read the message
                await asyncio.sleep(1)
                print("\nChat session ended.")
                break
            
            # Update QA dictionary based on the response
            updated = await update_qa_dictionary(agent, full_response, qa_dict, user_input)
            if updated:
                # Validate the QA dictionary before saving
                for entry in qa_dict:
                    for key, value in entry.items():
                        # Ensure attempts is always a number
                        if not isinstance(value.get("attempts"), int):
                            value["attempts"] = 0
                        
                        # Ensure attempts doesn't exceed 3
                        if value.get("attempts", 0) > 3:
                            value["attempts"] = 3
                        
                        # Clean and validate response values
                        if value.get("response") is not None:
                            if isinstance(value["response"], str):
                                # Remove any JSON or markdown artifacts
                                if "```" in value["response"]:
                                    value["response"] = value["response"].replace("```json", "").replace("```", "")
                                
                                # Remove any JSON characters
                                for char in ['{', '}', '[', ']', '"', "'", '`']:
                                    value["response"] = value["response"].replace(char, '')
                                
                                value["response"] = value["response"].strip()
                                
                                # Field-specific validation
                                if key == "age" and value["response"] != "refused":
                                    # Age should be numeric
                                    if not value["response"].isdigit():
                                        import re
                                        age_match = re.search(r'\b(\d+)\b', value["response"])
                                        if age_match:
                                            value["response"] = age_match.group(1)
                                        else:
                                            logger.warning(f"Invalid age value: {value['response']}, resetting to None")
                                            value["response"] = None
                                
                                # Handle empty strings
                                if not value["response"]:
                                    value["response"] = None
                
                # Save the updated dictionary with nice formatting
                logger.info(f"Saving updated QA dictionary: {qa_dict}")
                with open(qa_dict_path, 'w') as f:
                    json.dump(qa_dict, f, indent=2)
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error in chat loop: {e}")
        print(f"\nAn error occurred: {e}")
    
    # Save final state
    with open(qa_dict_path, 'w') as f:
        json.dump(qa_dict, f, indent=2)
    
    print("Chat session ended.")

if __name__ == "__main__":
    asyncio.run(main())