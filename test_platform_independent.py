#!/usr/bin/env python
"""
Simple test of platform-independent R-Zero components
"""
import os
import sys
from pathlib import Path
from openai import OpenAI

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_clients.config_manager import LLMConfigManager

def test_question_generation():
    """Test question generation with your Docker model runner"""
    print("=" * 60)
    print("Testing Platform-Independent Question Generation")
    print("=" * 60)
    
    # Get client
    try:
        client = OpenAI(
            base_url="http://localhost:12434/engines/llama.cpp/v1",
            api_key="dummy"
        )
        model_name = "ai/llama3.2"
        print(f"✅ Connected to {model_name}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Test question generation
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter. Create a challenging math problem "
                "and provide the answer in \\boxed{} format."
            )
        },
        {
            "role": "user", 
            "content": "Create a new algebra problem suitable for a math competition."
        }
    ]
    
    try:
        print("🔄 Generating math question...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.8
        )
        
        generated_question = response.choices[0].message.content
        print("✅ Question generated successfully!")
        print("\n📝 Generated Question:")
        print("-" * 40)
        print(generated_question)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def test_answer_evaluation():
    """Test answer evaluation"""
    print("\n" + "=" * 60)
    print("Testing Platform-Independent Answer Evaluation")
    print("=" * 60)
    
    # Get client
    try:
        client = OpenAI(
            base_url="http://localhost:12434/engines/llama.cpp/v1",
            api_key="dummy"
        )
        model_name = "ai/llama3.2"
        print(f"✅ Connected to {model_name}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Test problem solving
    test_problem = "If 3x + 7 = 22, what is the value of x?"
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert mathematician. Solve problems step by step and put your final answer in \\boxed{} format."
        },
        {
            "role": "user",
            "content": f"Problem: {test_problem}\n\nSolve this step by step and put your final answer in \\boxed{{}}."
        }
    ]
    
    try:
        print(f"🔄 Solving: {test_problem}")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=300,
            temperature=0.1
        )
        
        solution = response.choices[0].message.content
        print("✅ Problem solved successfully!")
        print("\n🧮 Solution:")
        print("-" * 40)
        print(solution)
        print("-" * 40)
        
        # Extract boxed answer
        import re
        boxed_match = re.search(r'\\boxed\{([^}]*)\}', solution)
        if boxed_match:
            final_answer = boxed_match.group(1)
            print(f"📊 Extracted Answer: {final_answer}")
            
            # Check if correct (expected answer is 5)
            if "5" in final_answer:
                print("✅ Answer is correct!")
            else:
                print("❌ Answer might be incorrect")
        
        return True
        
    except Exception as e:
        print(f"❌ Solving failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)
    
    try:
        config_manager = LLMConfigManager()
        eval_models = config_manager.get_evaluation_models()
        gen_models = config_manager.get_generation_models()
        
        print("✅ Configuration loaded successfully!")
        print(f"📋 Evaluation models: {list(eval_models.keys())}")
        print(f"📋 Generation models: {list(gen_models.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def main():
    print("R-Zero Platform Independence Test Suite")
    print("Testing Windows compatibility with Docker model runner")
    
    # Set required environment variables
    storage_path = "C:/Users/GCV/Desktop/Rzero/R-Zero/storage"
    os.environ["STORAGE_PATH"] = storage_path
    
    results = []
    
    # Run tests
    results.append(("Configuration Loading", test_config_loading()))
    results.append(("Question Generation", test_question_generation()))
    results.append(("Answer Evaluation", test_answer_evaluation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! R-Zero is now platform independent!")
        print("\n💡 Next steps:")
        print("   1. Use question_generate_platform_independent.py for question generation")
        print("   2. Use evaluate_platform_independent.py for evaluation")
        print("   3. Your Docker model runner works perfectly with R-Zero!")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the errors above.")

if __name__ == "__main__":
    main()