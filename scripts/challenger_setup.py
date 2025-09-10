#!/usr/bin/env python
"""
Challenger Model Setup Tool
Easy configuration and testing of external challenger models like GPT-4, Claude, etc.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_clients.config_manager import LLMConfigManager
from llm_clients.external_model_client import ChallengerModelManager, ExternalModelClient


def setup_openai_challenger():
    """Setup OpenAI GPT as challenger"""
    print("Setting up OpenAI GPT as challenger...")
    
    api_key = input("Enter your OpenAI API key (or press Enter to use OPENAI_API_KEY env var): ").strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: No API key provided and OPENAI_API_KEY not set")
            return False
    
    # Model selection
    models = [
        "gpt-4o",
        "gpt-4-turbo", 
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini"
    ]
    
    print("Available OpenAI models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    try:
        choice = int(input("Select model (1-6): ")) - 1
        model_name = models[choice]
    except (ValueError, IndexError):
        print("Invalid selection, using gpt-4o")
        model_name = "gpt-4o"
    
    config = {
        "provider": "openai",
        "config": {
            "base_url": "https://api.openai.com/v1",
            "model_name": model_name,
            "api_key": api_key,
            "max_tokens": 4096,
            "temperature": 0.7
        }
    }
    
    return test_and_save_config("challenger", config, "OpenAI GPT")


def setup_anthropic_challenger():
    """Setup Anthropic Claude as challenger"""
    print("Setting up Anthropic Claude as challenger...")
    
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic library not installed")
        print("Install with: pip install anthropic")
        return False
    
    api_key = input("Enter your Anthropic API key (or press Enter to use ANTHROPIC_API_KEY env var): ").strip()
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: No API key provided and ANTHROPIC_API_KEY not set")
            return False
    
    # Model selection
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]
    
    print("Available Claude models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    try:
        choice = int(input("Select model (1-5): ")) - 1
        model_name = models[choice]
    except (ValueError, IndexError):
        print("Invalid selection, using claude-3-5-sonnet-20241022")
        model_name = "claude-3-5-sonnet-20241022"
    
    config = {
        "provider": "anthropic",
        "config": {
            "model_name": model_name,
            "api_key": api_key,
            "max_tokens": 4096,
            "temperature": 0.7
        }
    }
    
    return test_and_save_config("challenger", config, "Anthropic Claude")


def setup_ollama_challenger():
    """Setup Ollama as challenger"""
    print("Setting up Ollama as challenger...")
    
    base_url = input("Enter Ollama URL (default: http://localhost:11434): ").strip()
    if not base_url:
        base_url = "http://localhost:11434"
    
    model_name = input("Enter model name (e.g., llama3.2, qwen2.5:14b): ").strip()
    if not model_name:
        print("Error: Model name is required")
        return False
    
    config = {
        "provider": "ollama",
        "config": {
            "base_url": f"{base_url}/v1",
            "model_name": model_name,
            "max_tokens": 4096,
            "temperature": 0.7
        }
    }
    
    return test_and_save_config("challenger", config, f"Ollama ({model_name})")


def setup_custom_challenger():
    """Setup custom OpenAI-compatible API as challenger"""
    print("Setting up custom OpenAI-compatible API as challenger...")
    
    base_url = input("Enter API base URL: ").strip()
    if not base_url:
        print("Error: Base URL is required")
        return False
    
    model_name = input("Enter model name: ").strip()
    if not model_name:
        print("Error: Model name is required")
        return False
    
    api_key = input("Enter API key (or 'dummy' for no auth): ").strip()
    if not api_key:
        api_key = "dummy"
    
    config = {
        "provider": "openai",
        "config": {
            "base_url": base_url,
            "model_name": model_name,
            "api_key": api_key,
            "max_tokens": 4096,
            "temperature": 0.7
        }
    }
    
    return test_and_save_config("challenger", config, f"Custom ({model_name})")


def test_and_save_config(config_key: str, config: dict, model_description: str) -> bool:
    """Test the configuration and save if successful"""
    print(f"\nTesting connection to {model_description}...")
    
    try:
        # Test the configuration
        client = ExternalModelClient(config["provider"], config["config"])
        if client.test_connection():
            print(f"✅ Connection successful to {model_description}")
            
            # Save configuration
            save_config(config_key, config)
            print(f"✅ Configuration saved successfully")
            return True
        else:
            print(f"❌ Connection failed to {model_description}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {model_description}: {e}")
        return False


def save_config(config_key: str, config: dict):
    """Save configuration to YAML file"""
    config_path = project_root / "config" / "llm_config.yaml"
    
    # Load existing config
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
    else:
        full_config = {}
    
    # Update the config
    if "generation_models" not in full_config:
        full_config["generation_models"] = {}
    
    full_config["generation_models"][config_key] = config
    
    # Save updated config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(full_config, f, default_flow_style=False, sort_keys=False)


def list_current_challengers():
    """List currently configured challengers"""
    try:
        config_manager = LLMConfigManager()
        challenger_manager = ChallengerModelManager(config_manager)
        
        challengers = challenger_manager.list_challengers()
        if challengers:
            print("\nCurrently configured challengers:")
            for name, model in challengers.items():
                print(f"  • {name}: {model}")
        else:
            print("\nNo challengers currently configured")
            
    except Exception as e:
        print(f"Error listing challengers: {e}")


def test_challenger_generation():
    """Test question generation with current challenger"""
    try:
        config_manager = LLMConfigManager()
        challenger_manager = ChallengerModelManager(config_manager)
        
        current = challenger_manager.get_current_challenger()
        if not current:
            print("No challenger model configured")
            return
        
        print("Testing question generation...")
        questions = challenger_manager.generate_challenging_questions(
            domain="mathematics",
            difficulty="medium", 
            num_questions=3
        )
        
        if questions:
            print(f"\n✅ Generated {len(questions)} questions:")
            for i, q in enumerate(questions, 1):
                print(f"\n{i}. Question: {q.get('question', 'N/A')}")
                print(f"   Answer: {q.get('answer', 'N/A')}")
        else:
            print("❌ No questions generated")
            
    except Exception as e:
        print(f"Error testing generation: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main interactive setup"""
    print("🤖 R-Zero Challenger Model Setup Tool")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Setup OpenAI GPT challenger")
        print("2. Setup Anthropic Claude challenger") 
        print("3. Setup Ollama challenger")
        print("4. Setup custom API challenger")
        print("5. List current challengers")
        print("6. Test question generation")
        print("7. Exit")
        
        try:
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                setup_openai_challenger()
            elif choice == "2":
                setup_anthropic_challenger()
            elif choice == "3":
                setup_ollama_challenger()
            elif choice == "4":
                setup_custom_challenger()
            elif choice == "5":
                list_current_challengers()
            elif choice == "6":
                test_challenger_generation()
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid option, please try again")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()