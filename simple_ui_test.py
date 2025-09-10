#!/usr/bin/env python
"""
Simple test for UI model selection functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_api_models():
    """Test the API model definitions"""
    try:
        from ui_training_system.real_training_api import ChallengerConfig, SolverConfig, TrainingConfigRequest
        
        # Test ChallengerConfig
        challenger_config = ChallengerConfig(
            mode="external",
            provider="openai", 
            model_name="gpt-4o",
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )
        
        # Test SolverConfig
        solver_config = SolverConfig(
            mode="custom",
            base_url="http://localhost:12434/engines/llama.cpp/v1",
            model_name="ai/llama3.2"
        )
        
        # Test TrainingConfigRequest
        training_config = TrainingConfigRequest(
            base_model="microsoft/DialoGPT-small",
            challenger_config=challenger_config,
            solver_config=solver_config,
            experiment_name="test_experiment"
        )
        
        print("SUCCESS: API models work correctly")
        return True
        
    except Exception as e:
        print(f"ERROR: API model test failed: {e}")
        return False


def test_html_structure():
    """Test HTML structure"""
    try:
        html_file = Path("ui_training_system/real_training_ui.html")
        
        if not html_file.exists():
            print("ERROR: HTML file not found")
            return False
            
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for key challenger elements
        if 'id="challengerProvider"' not in html_content:
            print("ERROR: Challenger provider selector missing")
            return False
            
        if 'id="challengerOpenAIModel"' not in html_content:
            print("ERROR: OpenAI model selector missing")
            return False
        
        if 'function getChallengerConfig' not in html_content:
            print("ERROR: getChallengerConfig function missing")
            return False
            
        print("SUCCESS: HTML structure is correct")
        return True
        
    except Exception as e:
        print(f"ERROR: HTML test failed: {e}")
        return False


def main():
    """Run tests"""
    print("Testing UI Model Selection")
    print("=" * 30)
    
    tests = [
        test_api_models,
        test_html_structure
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\nAll tests passed!")
        print("The UI model selection feature is working correctly.")
        print("\nTo use:")
        print("1. Start API: python ui_training_system/real_training_api.py")
        print("2. Open: ui_training_system/real_training_ui.html")
    else:
        print("\nSome tests failed. Check the errors above.")


if __name__ == "__main__":
    main()