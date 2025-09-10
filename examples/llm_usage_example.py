#!/usr/bin/env python3
"""
Example usage of the extended LLM clients for R-Zero framework

This script demonstrates how to:
1. Use Docker model runners
2. Use Ollama clients  
3. Use OpenAI-compatible APIs
4. Configure model providers via YAML config
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_clients import (
    DockerModelRunner, 
    OllamaClient, 
    OpenAICompatibleClient,
    ModelFactory,
    LLMConfigManager
)


def test_docker_model_runner():
    """Test Docker model runner"""
    print("=" * 50)
    print("Testing Docker Model Runner")
    print("=" * 50)
    
    client = DockerModelRunner(
        base_url="http://localhost:12434/engines/llama.cpp/v1",
        model_name="llama3.2"
    )
    
    print(f"Available: {client.is_available()}")
    
    if client.is_available():
        messages = [
            {"role": "user", "content": "What is 2+2? Give only the number."}
        ]
        response = client.generate_response(messages, temperature=0.1, max_tokens=10)
        print(f"Response: {response}")
    else:
        print("Docker model runner not available - make sure it's running on localhost:12434")


def test_ollama_client():
    """Test Ollama client"""
    print("\n" + "=" * 50)
    print("Testing Ollama Client")
    print("=" * 50)
    
    client = OllamaClient(
        base_url="http://localhost:11434",
        model_name="llama3.2"
    )
    
    print(f"Available: {client.is_available()}")
    
    if client.is_available():
        messages = [
            {"role": "user", "content": "What is 3+3? Give only the number."}
        ]
        response = client.generate_response(messages, temperature=0.1, max_tokens=10)
        print(f"Response: {response}")
    else:
        print("Ollama not available or model not loaded")
        print("Try: ollama pull llama3.2")


def test_openai_compatible():
    """Test OpenAI-compatible client"""
    print("\n" + "=" * 50)
    print("Testing OpenAI-Compatible Client")
    print("=" * 50)
    
    # Test with a hypothetical OpenAI-compatible endpoint
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",  # Example endpoint
        model_name="gpt-3.5-turbo"
    )
    
    print(f"Available: {client.is_available()}")
    
    if client.is_available():
        messages = [
            {"role": "user", "content": "What is 4+4? Give only the number."}
        ]
        response = client.generate_response(messages, temperature=0.1, max_tokens=10)
        print(f"Response: {response}")
    else:
        print("OpenAI-compatible endpoint not available")


def test_model_factory():
    """Test model factory with configuration"""
    print("\n" + "=" * 50)
    print("Testing Model Factory")
    print("=" * 50)
    
    # Test creating clients via factory
    configs = [
        {
            "name": "Docker Runner",
            "provider": "docker",
            "config": {
                "base_url": "http://localhost:12434/engines/llama.cpp/v1",
                "model_name": "llama3.2"
            }
        },
        {
            "name": "Ollama",
            "provider": "ollama", 
            "config": {
                "base_url": "http://localhost:11434",
                "model_name": "llama3.2"
            }
        }
    ]
    
    for config in configs:
        try:
            client = ModelFactory.create_client(config["provider"], config["config"])
            available = client.is_available()
            print(f"{config['name']}: Available = {available}")
        except Exception as e:
            print(f"{config['name']}: Error = {e}")


def test_config_manager():
    """Test configuration manager"""
    print("\n" + "=" * 50)
    print("Testing Configuration Manager")
    print("=" * 50)
    
    config_manager = LLMConfigManager()
    
    print("Evaluation Models:")
    eval_models = config_manager.get_evaluation_models()
    if "primary" in eval_models:
        primary = eval_models["primary"]
        print(f"  Primary: {primary['provider']} - {primary['config'].get('model_name')}")
    
    fallbacks = eval_models.get("fallback", [])
    if fallbacks:
        print("  Fallbacks:")
        for i, fallback in enumerate(fallbacks, 1):
            print(f"    {i}. {fallback['provider']} - {fallback['config'].get('model_name')}")
    
    print("\nGeneration Models:")
    gen_models = config_manager.get_generation_models()
    for role, model_config in gen_models.items():
        print(f"  {role}: {model_config['provider']} - {model_config['config'].get('model_name')}")


def main():
    print("R-Zero Extended LLM Clients Example")
    print("This script tests various model providers for the R-Zero framework")
    
    # Test individual clients
    test_docker_model_runner()
    test_ollama_client()
    test_openai_compatible()
    
    # Test factory and configuration
    test_model_factory()
    test_config_manager()
    
    print("\n" + "=" * 50)
    print("Usage Summary:")
    print("=" * 50)
    print("1. Docker Model Runner: Supports llama.cpp-based containers")
    print("2. Ollama Client: Works with local Ollama installations")
    print("3. OpenAI Compatible: Works with any OpenAI-compatible API")
    print("4. Model Factory: Create clients from configuration")
    print("5. Config Manager: Manages YAML-based configuration")
    
    print("\nTo use in evaluation:")
    print("python evaluation/results_recheck_extended.py --model_name YourModel")


if __name__ == "__main__":
    main()