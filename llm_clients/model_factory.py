"""
Model factory for creating LLM clients based on configuration
"""
from typing import Dict, Any, Optional
from .base_client import BaseLLMClient
from .openai_compatible_client import OpenAICompatibleClient
from .docker_model_runner import DockerModelRunner
from .ollama_client import OllamaClient


class ModelFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create_client(provider: str, config: Dict[str, Any]) -> BaseLLMClient:
        """
        Create an LLM client based on provider type and configuration
        
        Args:
            provider: Type of provider ("openai", "docker", "ollama")
            config: Configuration dictionary with provider-specific settings
            
        Returns:
            BaseLLMClient instance
        """
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAICompatibleClient(
                base_url=config.get("base_url", "https://api.openai.com/v1"),
                model_name=config.get("model_name", "gpt-4o"),
                api_key=config.get("api_key")
            )
        
        elif provider == "docker":
            return DockerModelRunner(
                base_url=config.get("base_url", "http://localhost:12434/engines/llama.cpp/v1"),
                model_name=config.get("model_name", "llama3.2")
            )
        
        elif provider == "ollama":
            return OllamaClient(
                base_url=config.get("base_url", "http://localhost:11434"),
                model_name=config.get("model_name", "llama3.2")
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> Dict[str, str]:
        """Get list of available providers with descriptions"""
        return {
            "openai": "OpenAI-compatible API endpoints",
            "docker": "Docker-hosted model runners with llama.cpp backend",
            "ollama": "Ollama-hosted local models"
        }