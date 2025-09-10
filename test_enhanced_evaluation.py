#!/usr/bin/env python
"""
Test script for the enhanced evaluation system
Demonstrates improved evaluation with challenger models and custom datasets
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.enhanced_evaluator import EnhancedEvaluator
from llm_clients.config_manager import LLMConfigManager


def create_sample_dataset():
    """Create a sample custom dataset for testing"""
    sample_questions = [
        {
            "question": "A triangle has sides of length 3, 4, and 5. What is its area?",
            "answer": "6",
            "domain": "geometry",
            "difficulty": "medium"
        },
        {
            "question": "If f(x) = 2x + 3, what is f(5)?",
            "answer": "13", 
            "domain": "algebra",
            "difficulty": "easy"
        },
        {
            "question": "What is the derivative of x^3 + 2x^2 - 5x + 1?",
            "answer": "3x^2 + 4x - 5",
            "domain": "calculus", 
            "difficulty": "hard"
        },
        {
            "question": "A box contains 5 red balls and 3 blue balls. What is the probability of drawing a red ball?",
            "answer": "5/8",
            "domain": "probability",
            "difficulty": "medium"
        },
        {
            "question": "Solve for x: 2x + 7 = 15",
            "answer": "4",
            "domain": "algebra",
            "difficulty": "easy"
        }
    ]
    
    dataset_path = Path("storage/test_datasets/sample_math_dataset.json")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(sample_questions, f, indent=2)
    
    print(f"Created sample dataset: {dataset_path}")
    return str(dataset_path)


def test_challenger_connection():
    """Test challenger model connection"""
    print("=== Testing Challenger Model Connection ===")
    
    try:
        config_manager = LLMConfigManager()
        from llm_clients.external_model_client import ChallengerModelManager
        
        challenger_manager = ChallengerModelManager(config_manager)
        challengers = challenger_manager.list_challengers()
        
        if challengers:
            print("Available challengers:")
            for name, model in challengers.items():
                print(f"  • {name}: {model}")
                
            # Test connection to primary challenger
            current = challenger_manager.get_current_challenger()
            if current:
                print("\nTesting connection...")
                if current.test_connection():
                    print("✅ Challenger connection successful")
                    return True
                else:
                    print("❌ Challenger connection failed")
                    return False
            else:
                print("❌ No challenger model available")
                return False
        else:
            print("No challengers configured")
            print("\nTo setup a challenger, run: python scripts/challenger_setup.py")
            return False
            
    except Exception as e:
        print(f"Error testing challenger: {e}")
        return False


def test_question_generation():
    """Test question generation with challenger"""
    print("\n=== Testing Question Generation ===")
    
    try:
        evaluator = EnhancedEvaluator()
        
        # Generate a few test questions
        questions = evaluator.generate_questions_with_challenger(
            domain="mathematics",
            difficulty="medium",
            num_questions=3
        )
        
        if questions:
            print(f"✅ Generated {len(questions)} questions:")
            for i, q in enumerate(questions, 1):
                print(f"\n{i}. Q: {q.get('question', 'N/A')}")
                print(f"   A: {q.get('answer', 'N/A')}")
            return True
        else:
            print("❌ No questions generated")
            return False
            
    except Exception as e:
        print(f"Error generating questions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_dataset_loading():
    """Test custom dataset loading"""
    print("\n=== Testing Custom Dataset Loading ===")
    
    try:
        # Create and load sample dataset
        dataset_path = create_sample_dataset()
        
        evaluator = EnhancedEvaluator()
        dataset = evaluator.load_custom_dataset(dataset_path)
        normalized = evaluator.normalize_dataset(dataset)
        
        print(f"✅ Loaded {len(normalized)} questions from custom dataset:")
        for i, item in enumerate(normalized[:3], 1):
            print(f"\n{i}. Q: {item['question']}")
            print(f"   A: {item['answer']}")
            print(f"   Domain: {item['domain']}, Difficulty: {item['difficulty']}")
        
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False


def test_solver_evaluation():
    """Test solver evaluation"""
    print("\n=== Testing Solver Evaluation ===")
    
    # Simple test questions
    test_questions = [
        {"question": "What is 5 + 3?", "answer": "8"},
        {"question": "Calculate 7 * 6", "answer": "42"},
        {"question": "What is 15 - 9?", "answer": "6"}
    ]
    
    try:
        evaluator = EnhancedEvaluator()
        results = evaluator.evaluate_with_solver(test_questions, num_samples=2)
        
        if "error" in results:
            print(f"❌ Solver evaluation failed: {results['error']}")
            return False
        
        stats = results.get("statistics", {})
        print(f"✅ Solver evaluation completed:")
        print(f"   Overall accuracy: {stats.get('overall_accuracy', 0):.3f}")
        print(f"   Total questions: {stats.get('total_questions', 0)}")
        print(f"   Total correct: {stats.get('total_correct', 0)}")
        
        return True
        
    except Exception as e:
        print(f"Error in solver evaluation: {e}")
        return False


def test_comprehensive_evaluation():
    """Test comprehensive evaluation with both custom dataset and generated questions"""
    print("\n=== Testing Comprehensive Evaluation ===")
    
    try:
        # Create sample dataset
        dataset_path = create_sample_dataset()
        
        evaluator = EnhancedEvaluator()
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation(
            custom_dataset_path=dataset_path,
            num_generated_questions=2,  # Small number for testing
            num_samples_per_question=2,  # Small number for testing
            domain="mathematics",
            difficulty="medium"
        )
        
        # Save results
        evaluator.save_results(results, "test_comprehensive_evaluation.json")
        
        stats = results.get("statistics", {})
        print(f"✅ Comprehensive evaluation completed:")
        print(f"   Overall accuracy: {stats.get('overall_accuracy', 0):.3f}")
        print(f"   Average difficulty: {stats.get('average_difficulty_score', 0):.3f}")
        print(f"   Average reasoning quality: {stats.get('average_reasoning_quality', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error in comprehensive evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 Enhanced Evaluation System Test Suite")
    print("=" * 50)
    
    # Set up storage path if not set
    if not os.getenv("STORAGE_PATH"):
        os.environ["STORAGE_PATH"] = "storage"
    
    tests = [
        ("Challenger Connection", test_challenger_connection),
        ("Question Generation", test_question_generation), 
        ("Custom Dataset Loading", test_custom_dataset_loading),
        ("Solver Evaluation", test_solver_evaluation),
        ("Comprehensive Evaluation", test_comprehensive_evaluation)
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
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced evaluation system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        
        if passed == 0:
            print("\nQuick setup guide:")
            print("1. Set up your challenger model: python scripts/challenger_setup.py")
            print("2. Make sure your solver model is running")
            print("3. Set STORAGE_PATH environment variable if needed")


if __name__ == "__main__":
    main()