#!/usr/bin/env python
"""
Setup script to configure LangSmith integration for R-Zero
"""

import os
from pathlib import Path


def setup_environment():
    """Setup environment variables for LangSmith"""
    
    print("🔧 Setting up LangSmith for R-Zero Training")
    print("=" * 40)
    
    # Your LangSmith API token
    langsmith_token = "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"
    
    print(f"✅ LangSmith API Token: {langsmith_token[:20]}...")
    
    # Set environment variables
    os.environ['LANGSMITH_API_KEY'] = langsmith_token
    os.environ['LANGSMITH_PROJECT'] = 'rzero-training'
    
    # Also set storage path if not set
    if not os.environ.get('STORAGE_PATH'):
        os.environ['STORAGE_PATH'] = 'storage'
    
    print("✅ Environment variables set:")
    print(f"   LANGSMITH_API_KEY: {os.environ['LANGSMITH_API_KEY'][:20]}...")
    print(f"   LANGSMITH_PROJECT: {os.environ.get('LANGSMITH_PROJECT')}")
    print(f"   STORAGE_PATH: {os.environ.get('STORAGE_PATH')}")
    
    return True


def create_env_file():
    """Create a .env file for persistent configuration"""
    
    env_content = f"""# LangSmith Configuration for R-Zero
LANGSMITH_API_KEY=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
LANGSMITH_PROJECT=rzero-training

# Storage Configuration
STORAGE_PATH=storage

# Optional: OpenAI API Key (if using GPT models)
# OPENAI_API_KEY=your-openai-key-here

# Optional: Anthropic API Key (if using Claude models) 
# ANTHROPIC_API_KEY=your-anthropic-key-here
"""
    
    env_file = Path(".env")
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Created {env_file} with LangSmith configuration")
        print("   You can edit this file to add other API keys as needed.")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def check_langsmith_dependency():
    """Check if LangSmith is installed"""
    
    try:
        import langsmith
        print("✅ LangSmith library is installed")
        return True
    except ImportError:
        print("❌ LangSmith library not found")
        print("   Install with: pip install langsmith")
        return False


def main():
    """Setup LangSmith integration"""
    
    print("🚀 LangSmith Setup for R-Zero Training System")
    print("=" * 50)
    
    # Check dependency
    if not check_langsmith_dependency():
        print("\nPlease install LangSmith first:")
        print("pip install langsmith")
        return
    
    # Setup environment
    setup_environment()
    
    # Create .env file
    create_env_file()
    
    print("\n🎉 LangSmith setup completed successfully!")
    print("\nNext steps:")
    print("1. Your training experiments will be automatically tracked")
    print("2. View experiments at: https://smith.langchain.com/")
    print("3. Run training as usual - LangSmith logging is now enabled")
    
    print("\nTest the integration:")
    print("python test_langsmith_integration.py")


if __name__ == "__main__":
    main()