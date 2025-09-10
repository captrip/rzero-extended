"""
OpenAI-compatible API client for Docker model runners and other OpenAI-compatible endpoints
"""
import requests
import json
from typing import Dict, Any, Optional, List
from .base_client import BaseLLMClient


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs (like Docker model runners)"""
    
    def __init__(self, base_url: str, model_name: str, api_key: Optional[str] = None):
        super().__init__(base_url, model_name, api_key)
        # Ensure the base_url ends with the correct path
        if not self.base_url.endswith('/chat/completions'):
            if self.base_url.endswith('/'):
                self.base_url = self.base_url + 'chat/completions'
            else:
                self.base_url = self.base_url + '/chat/completions'
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate a response using OpenAI-compatible API"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenAI-compatible API: {e}")
            return "Error: Failed to get response"
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            return "Error: Invalid response format"
    
    def is_available(self) -> bool:
        """Check if the endpoint is available"""
        try:
            # Try a simple health check by making a minimal request
            test_messages = [{"role": "user", "content": "test"}]
            response = self.generate_response(test_messages, max_tokens=1)
            return not response.startswith("Error:")
        except Exception:
            return False