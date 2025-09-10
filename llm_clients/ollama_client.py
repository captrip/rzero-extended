"""
Ollama client for R-Zero framework
"""
import requests
import json
from typing import Dict, Any, Optional, List
from .base_client import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Client for Ollama-hosted models"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        super().__init__(base_url, model_name, api_key=None)
        # Ollama uses different endpoints
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.models_url = f"{self.base_url}/api/tags"
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate a response using Ollama API"""
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.chat_url,
                headers=headers,
                json=payload,
                timeout=60  # Ollama can be slower than API services
            )
            response.raise_for_status()
            
            result = response.json()
            return result['message']['content']
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return "Error: Failed to get response"
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            return "Error: Invalid response format"
    
    def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded"""
        try:
            response = requests.get(self.models_url, timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                return self.model_name in available_models
            return False
        except Exception as e:
            print(f"Ollama not available: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull/download the model if not available"""
        try:
            pull_url = f"{self.base_url}/api/pull"
            payload = {"name": self.model_name}
            
            response = requests.post(pull_url, json=payload, timeout=300)  # 5 minutes timeout
            return response.status_code == 200
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False