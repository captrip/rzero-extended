"""
Base client interface for LLM providers
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, base_url: str, model_name: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model endpoint is available"""
        pass