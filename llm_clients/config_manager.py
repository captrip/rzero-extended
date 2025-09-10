"""
Configuration manager for LLM clients
"""
import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class LLMConfigManager:
    """Manages LLM configuration for the R-Zero framework"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config file in the project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "llm_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Replace environment variables
            config = self._replace_env_vars(config)
            return config
            
        except FileNotFoundError:
            print(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _replace_env_vars(self, obj: Any) -> Any:
        """Replace ${VAR_NAME} patterns with environment variables"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.getenv(var_name, obj)  # Return original if env var not found
        else:
            return obj
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "evaluation_models": {
                "primary": {
                    "provider": "ollama",
                    "config": {
                        "base_url": "http://localhost:11434",
                        "model_name": "llama3.2"
                    }
                },
                "fallback": [
                    {
                        "provider": "docker",
                        "config": {
                            "base_url": "http://localhost:12434/engines/llama.cpp/v1",
                            "model_name": "llama3.2"
                        }
                    }
                ]
            },
            "generation_models": {
                "challenger": {
                    "provider": "docker",
                    "config": {
                        "base_url": "http://localhost:12434/engines/llama.cpp/v1",
                        "model_name": "llama3.2"
                    }
                }
            },
            "timeouts": {
                "evaluation": 30,
                "generation": 60
            },
            "retry": {
                "max_attempts": 3,
                "backoff_factor": 2.0
            }
        }
    
    def get_evaluation_models(self) -> Dict[str, Any]:
        """Get evaluation model configurations"""
        return self.config.get("evaluation_models", {})
    
    def get_generation_models(self) -> Dict[str, Any]:
        """Get generation model configurations"""
        return self.config.get("generation_models", {})
    
    def get_timeouts(self) -> Dict[str, int]:
        """Get timeout settings"""
        return self.config.get("timeouts", {"evaluation": 30, "generation": 60})
    
    def get_retry_settings(self) -> Dict[str, Any]:
        """Get retry settings"""
        return self.config.get("retry", {"max_attempts": 3, "backoff_factor": 2.0})