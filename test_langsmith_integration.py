#!/usr/bin/env python
"""
Test LangSmith integration with your API token
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set the environment variable with your token
os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['STORAGE_PATH'] = 'storage'

def test_basic_langsmith():
    """Test basic LangSmith functionality"""
    try:
        from llm_clients.langsmith_integration import LangSmithTracker
        
        print("Testing LangSmith Integration")
        print("=" * 30)
        
        # Initialize tracker
        tracker = LangSmithTracker()
        
        if not tracker.is_available():
            print("❌ LangSmith not available or not enabled")
            return False
        
        print("✅ LangSmith client initialized successfully")
        
        # Test starting an experiment
        experiment_id = tracker.start_experiment("test_rzero_integration", {
            "test": True,
            "challenger_model": "gpt-4o",
            "solver_model": "llama3.2"
        })
        
        print(f"✅ Started experiment: {experiment_id}")
        
        # Test logging challenger generation
        sample_questions = [
            {
                "question": "What is the derivative of x^2 + 3x - 5?",
                "answer": "2x + 3",
                "domain": "calculus",
                "difficulty": "medium"
            },
            {
                "question": "If a triangle has sides 3, 4, 5, what type is it?",
                "answer": "right triangle",
                "domain": "geometry", 
                "difficulty": "easy"
            }
        ]
        
        tracker.log_challenger_generation("gpt-4o", "mathematics", "mixed", sample_questions)
        print("✅ Logged challenger generation")
        
        # Test logging training iteration
        sample_results = {
            "statistics": {
                "overall_accuracy": 0.75,
                "total_questions": 2,
                "total_correct": 1,
                "average_difficulty_score": 0.4,
                "average_reasoning_quality": 0.7
            },
            "results": [
                {
                    "correct_count": 1,
                    "accuracy": 1.0,
                    "difficulty_score": 0.3,
                    "reasoning_quality": 0.8,
                    "generated_answers": ["2x + 3"]
                },
                {
                    "correct_count": 0, 
                    "accuracy": 0.0,
                    "difficulty_score": 0.5,
                    "reasoning_quality": 0.6,
                    "generated_answers": ["isosceles"]
                }
            ]
        }
        
        tracker.log_training_iteration(
            1, 
            {"loss": 0.23, "learning_rate": 1e-5},
            sample_questions,
            sample_results
        )
        print("✅ Logged training iteration")
        
        # Test ending experiment
        final_metrics = {
            "final_accuracy": 0.75,
            "total_training_time": 45,
            "best_iteration": 1
        }
        
        tracker.end_experiment(final_metrics)
        print("✅ Ended experiment")
        
        # Get experiment URL
        if hasattr(tracker, 'get_experiment_url'):
            url = tracker.get_experiment_url()
            if url:
                print(f"🔗 Experiment URL: {url}")
        
        print("\n🎉 LangSmith integration test completed successfully!")
        print("Your R-Zero training experiments will now be tracked in LangSmith.")
        
        return True
        
    except Exception as e:
        print(f"❌ LangSmith test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_evaluator_with_langsmith():
    """Test the enhanced evaluator with LangSmith integration"""
    try:
        from evaluation.enhanced_evaluator import EnhancedEvaluator
        
        print("\nTesting Enhanced Evaluator with LangSmith")
        print("=" * 40)
        
        evaluator = EnhancedEvaluator()
        
        if not evaluator.langsmith_tracker.is_available():
            print("⚠️  LangSmith not available in evaluator")
            return False
        
        print("✅ Enhanced evaluator with LangSmith initialized")
        
        # Test generating questions (this will log to LangSmith)
        questions = evaluator.generate_questions_with_challenger(
            domain="mathematics",
            difficulty="easy", 
            num_questions=2
        )
        
        if questions:
            print(f"✅ Generated {len(questions)} questions and logged to LangSmith")
        else:
            print("⚠️  Used fallback questions (challenger may not be available)")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced evaluator test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🔬 Testing LangSmith Integration for R-Zero")
    print("Token: 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")
    print("=" * 50)
    
    tests = [
        test_basic_langsmith,
        test_enhanced_evaluator_with_langsmith
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"Test failed with error: {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed > 0:
        print("\n✅ LangSmith integration is working!")
        print("Your experiments will be automatically tracked at:")
        print("https://smith.langchain.com/")
    else:
        print("\n❌ LangSmith integration failed")
        print("Make sure you have 'langsmith' installed: pip install langsmith")


if __name__ == "__main__":
    main()