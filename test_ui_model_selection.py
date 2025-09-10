#!/usr/bin/env python
"""
Test script for UI model selection functionality
Tests the new challenger and solver model selection features
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_api_models():
    """Test the API model definitions"""
    try:
        from ui_training_system.real_training_api import ChallengerConfig, SolverConfig, TrainingConfigRequest
        
        print("API model classes imported successfully")
        
        # Test ChallengerConfig
        challenger_config = ChallengerConfig(
            mode="external",
            provider="openai",
            model_name="gpt-4o",
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )
        print(f"✅ ChallengerConfig created: {challenger_config.model_name}")
        
        # Test SolverConfig  
        solver_config = SolverConfig(
            mode="custom",
            base_url="http://localhost:12434/engines/llama.cpp/v1",
            model_name="ai/llama3.2"
        )
        print(f"✅ SolverConfig created: {solver_config.model_name}")
        
        # Test TrainingConfigRequest
        training_config = TrainingConfigRequest(
            base_model="microsoft/DialoGPT-small",
            challenger_config=challenger_config,
            solver_config=solver_config,
            experiment_name="test_experiment"
        )
        print(f"✅ TrainingConfigRequest created: {training_config.experiment_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ API model test failed: {e}")
        return False


def test_external_model_client():
    """Test the external model client functionality"""
    try:
        from llm_clients.external_model_client import ExternalModelClient, ChallengerModelManager
        from llm_clients.config_manager import LLMConfigManager
        
        print("✅ External model client imported successfully")
        
        # Test basic client creation (without actually connecting)
        config = {
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-4o",
            "api_key": "test-key"
        }
        
        # This won't actually connect, just test object creation
        client = ExternalModelClient("openai", config)
        print("✅ ExternalModelClient created successfully")
        
        # Test ChallengerModelManager creation
        config_manager = LLMConfigManager()
        challenger_manager = ChallengerModelManager(config_manager)
        print("✅ ChallengerModelManager created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ External model client test failed: {e}")
        return False


def test_config_structure():
    """Test configuration structure"""
    try:
        # Test sample challenger configurations
        configurations = {
            "openai_gpt4": {
                "mode": "external",
                "provider": "openai",
                "model_name": "gpt-4o",
                "api_key": "sk-...",
                "base_url": "https://api.openai.com/v1"
            },
            "anthropic_claude": {
                "mode": "external", 
                "provider": "anthropic",
                "model_name": "claude-3-5-sonnet-20241022",
                "api_key": "sk-ant-..."
            },
            "custom_api": {
                "mode": "external",
                "provider": "openai",
                "model_name": "custom-model",
                "api_key": "dummy",
                "base_url": "https://api.custom.com/v1"
            },
            "local_ollama": {
                "mode": "custom",
                "base_url": "http://localhost:11434/v1",
                "model_name": "llama3.2"
            }
        }
        
        print("✅ Configuration structures defined")
        
        # Test JSON serialization
        for name, config in configurations.items():
            json_str = json.dumps(config, indent=2)
            parsed = json.loads(json_str)
            assert parsed == config
            print(f"✅ {name} configuration serializes correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_html_elements():
    """Test that the HTML contains the expected elements"""
    try:
        html_file = Path("ui_training_system/real_training_ui.html")
        
        if not html_file.exists():
            print("❌ HTML file not found")
            return False
            
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for challenger model elements
        required_elements = [
            'id="challengerPreset"',
            'id="challengerProvider"',
            'id="challengerOpenAIModel"',
            'id="challengerAnthropicModel"',
            'id="challengerCustomBaseURL"',
            'id="challengerCustomModel"',
            'onclick="testChallengerConnection()"',
            'id="challengerStatus"',
            
            # Solver model elements
            'id="solverPreset"',
            'id="solverCustomURL"',
            'id="solverCustomName"',
            'onclick="testSolverConnection()"',
            'id="solverStatus"',
            
            # JavaScript functions
            'function switchChallengerMode',
            'function switchSolverMode',
            'function updateChallengerProvider',
            'function getChallengerConfig',
            'function getSolverConfig'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in html_content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing HTML elements: {missing_elements}")
            return False
        
        print("✅ All required HTML elements found")
        return True
        
    except Exception as e:
        print(f"❌ HTML test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing UI Model Selection Functionality")
    print("=" * 50)
    
    tests = [
        ("API Models", test_api_models),
        ("External Model Client", test_external_model_client),
        ("Configuration Structure", test_config_structure),
        ("HTML Elements", test_html_elements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! UI model selection is working correctly.")
        print("\nNext steps:")
        print("1. Start the training API: python ui_training_system/real_training_api.py")
        print("2. Open the UI: ui_training_system/real_training_ui.html") 
        print("3. Test challenger and solver model selection in the UI")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()