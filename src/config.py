"""
Configuration management module for the chat application.
Loads configuration from docker/.env file.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

def load_config() -> Dict[str, Any]:
    """
    Load configuration from docker/.env file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Try to load from docker/.env first
    docker_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker", ".env")
    
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