"""
Enhanced external model client for challenger models
Supports OpenAI, Anthropic, and other external providers
"""

import os
import time
from typing import Dict, List, Optional, Any
from openai import OpenAI
import requests
import json

try:
    import anthropic
except ImportError:
    anthropic = None


class ExternalModelClient:
    """Universal client for external model providers"""
    
    def __init__(self, provider: str, config: Dict[str, Any]):
        self.provider = provider.lower()
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider"""
        if self.provider == "openai":
            self.client = OpenAI(
                base_url=self.config.get("base_url", "https://api.openai.com/v1"),
                api_key=self.config.get("api_key", os.getenv("OPENAI_API_KEY"))
            )
        elif self.provider == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic library not installed. Install with: pip install anthropic")
            self.client = anthropic.Anthropic(
                api_key=self.config.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
            )
        elif self.provider == "ollama":
            self.client = OpenAI(
                base_url=self.config.get("base_url", "http://localhost:11434/v1"),
                api_key="dummy"  # Ollama doesn't need an API key
            )
        else:
            # Fallback to OpenAI-compatible API
            self.client = OpenAI(
                base_url=self.config.get("base_url"),
                api_key=self.config.get("api_key", "dummy")
            )
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response using the configured model"""
        model_name = self.config["model_name"]
        max_tokens = self.config.get("max_tokens", 2048)
        temperature = self.config.get("temperature", 0.7)
        
        # Override with any provided kwargs
        max_tokens = kwargs.get("max_tokens", max_tokens)
        temperature = kwargs.get("temperature", temperature)
        
        if self.provider == "anthropic":
            return self._generate_anthropic(messages, model_name, max_tokens, temperature)
        else:
            return self._generate_openai_compatible(messages, model_name, max_tokens, temperature)
    
    def _generate_openai_compatible(self, messages: List[Dict[str, str]], model_name: str, 
                                   max_tokens: int, temperature: float) -> str:
        """Generate response using OpenAI-compatible API"""
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {e}"
    
    def _generate_anthropic(self, messages: List[Dict[str, str]], model_name: str,
                           max_tokens: int, temperature: float) -> str:
        """Generate response using Anthropic Claude API"""
        try:
            # Convert OpenAI-style messages to Anthropic format
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            # Combine user messages if multiple
            if len(user_messages) == 1:
                user_content = user_messages[0]["content"]
            else:
                user_content = "\n\n".join([msg["content"] for msg in user_messages])
            
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message if system_message else None,
                messages=[{"role": "user", "content": user_content}]
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error generating Anthropic response: {e}")
            return f"Error: {e}"
    
    def test_connection(self) -> bool:
        """Test if the model client is working"""
        try:
            test_messages = [
                {"role": "user", "content": "Hello, please respond with 'OK' if you can see this message."}
            ]
            response = self.generate_response(test_messages, max_tokens=10, temperature=0)
            return "OK" in response or "ok" in response.lower()
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


class ChallengerModelManager:
    """Manages challenger model selection and usage"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.current_challenger = None
        self.available_challengers = {}
        self._load_challengers()
    
    def _load_challengers(self):
        """Load all available challenger configurations"""
        generation_models = self.config_manager.get_generation_models()
        
        # Load primary challenger
        if "challenger" in generation_models:
            config = generation_models["challenger"]
            try:
                client = ExternalModelClient(config["provider"], config["config"])
                self.available_challengers["primary"] = {
                    "client": client,
                    "config": config,
                    "name": config["config"]["model_name"]
                }
                self.current_challenger = "primary"
            except Exception as e:
                print(f"Failed to load primary challenger: {e}")
        
        # Load alternative challengers
        for key, config in generation_models.items():
            if key.startswith("challenger_") and key != "challenger":
                try:
                    client = ExternalModelClient(config["provider"], config["config"])
                    challenger_name = key.replace("challenger_", "")
                    self.available_challengers[challenger_name] = {
                        "client": client,
                        "config": config,
                        "name": config["config"]["model_name"]
                    }
                except Exception as e:
                    print(f"Failed to load challenger {key}: {e}")
    
    def list_challengers(self) -> Dict[str, str]:
        """List all available challenger models"""
        return {name: info["name"] for name, info in self.available_challengers.items()}
    
    def set_challenger(self, challenger_name: str) -> bool:
        """Set the active challenger model"""
        if challenger_name in self.available_challengers:
            # Test the connection first
            if self.available_challengers[challenger_name]["client"].test_connection():
                self.current_challenger = challenger_name
                print(f"Switched to challenger: {self.available_challengers[challenger_name]['name']}")
                return True
            else:
                print(f"Failed to connect to challenger: {challenger_name}")
                return False
        else:
            print(f"Challenger not found: {challenger_name}")
            return False
    
    def get_current_challenger(self) -> Optional[ExternalModelClient]:
        """Get the current challenger client"""
        if self.current_challenger and self.current_challenger in self.available_challengers:
            return self.available_challengers[self.current_challenger]["client"]
        return None
    
    def generate_challenging_questions(self, domain: str, difficulty: str, num_questions: int) -> List[Dict[str, str]]:
        """Generate challenging questions using the current challenger model"""
        challenger = self.get_current_challenger()
        if not challenger:
            raise RuntimeError("No challenger model available")
        
        # Enhanced prompt for better question generation
        system_prompt = f"""You are an expert in {domain} and specialize in creating challenging, thought-provoking questions that test deep understanding and problem-solving skills.

Your task is to generate {num_questions} high-quality questions at {difficulty} difficulty level.

Each question should:
1. Test conceptual understanding, not just memorization
2. Require multi-step reasoning
3. Have a clear, unambiguous correct answer
4. Be appropriate for the specified difficulty level
5. Cover different aspects of the domain

Format each question as a JSON object with 'question' and 'answer' fields."""

        user_prompt = f"""Generate {num_questions} challenging {difficulty}-level questions in {domain}.

Requirements:
- Each question should be unique and test different concepts
- Provide the correct answer for each question
- Make sure answers are concise but complete
- Vary the question types (calculation, conceptual, application, etc.)

Return the questions as a JSON array where each item has 'question' and 'answer' fields."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = challenger.generate_response(
                messages, 
                max_tokens=4000, 
                temperature=0.8  # Higher temperature for more diverse questions
            )
            
            # Try to parse JSON response
            try:
                import json
                questions = json.loads(response)
                if isinstance(questions, list):
                    return questions
                elif isinstance(questions, dict) and "questions" in questions:
                    return questions["questions"]
            except json.JSONDecodeError:
                pass
            
            # Fallback: try to extract questions from text
            return self._extract_questions_from_text(response)
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def _extract_questions_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract questions from unstructured text response"""
        questions = []
        lines = text.split('\n')
        current_question = None
        current_answer = None
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith(('q:', 'question:', 'q.', 'question ')):
                if current_question and current_answer:
                    questions.append({"question": current_question, "answer": current_answer})
                current_question = line.split(':', 1)[1].strip() if ':' in line else line
                current_answer = None
            elif line.lower().startswith(('a:', 'answer:', 'a.', 'answer ')):
                current_answer = line.split(':', 1)[1].strip() if ':' in line else line
        
        if current_question and current_answer:
            questions.append({"question": current_question, "answer": current_answer})
        
        return questions