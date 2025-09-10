"""
Docker Model Runner client for R-Zero framework
Supports llama.cpp-based Docker containers with OpenAI-compatible API
"""
from .openai_compatible_client import OpenAICompatibleClient


class DockerModelRunner(OpenAICompatibleClient):
    """
    Client for Docker-hosted model runners with llama.cpp backend
    
    Usage example:
        client = DockerModelRunner(
            base_url="http://localhost:12434/engines/llama.cpp/v1",
            model_name="llama3.2"
        )
    """
    
    def __init__(self, base_url: str = "http://localhost:12434/engines/llama.cpp/v1", model_name: str = "llama3.2"):
        # The base_url should already include the engine path
        super().__init__(base_url, model_name, api_key=None)
    
    def is_available(self) -> bool:
        """Check if the Docker model runner is available"""
        try:
            import requests
            # Check if the base endpoint is responding
            health_url = self.base_url.replace('/chat/completions', '/models')
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Docker model runner not available: {e}")
            return False