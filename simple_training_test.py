#!/usr/bin/env python3
"""
Simple test version of platform-independent training
Tests the core functionality without heavy model training
"""

import os
import sys
import json
import time
from pathlib import Path

# Set environment variables
os.environ["STORAGE_PATH"] = "C:/Users/GCV/Desktop/Rzero/R-Zero/storage"
os.environ["HUGGINGFACENAME"] = "test-user"

def test_question_generation():
    """Test question generation with Docker model runner"""
    print("=== Testing Question Generation ===")
    
    try:
        from question_generate.question_generate_platform_independent import generate_questions_simple
        
        output_file = "storage/generated_question/test_training_questions.json"
        
        print("Generating 10 test questions...")
        questions = generate_questions_simple(
            model_path="ai/llama3.2",
            num_questions=10,
            output_file=output_file
        )
        
        print(f"OK Generated {len(questions)} questions")
        for i, q in enumerate(questions[:3]):
            print(f"  {i+1}. {q.get('question', 'N/A')}")
            
        return True, questions
        
    except Exception as e:
        print(f"ERROR Question generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_question_evaluation():
    """Test question evaluation with Docker model runner"""
    print("\n=== Testing Question Evaluation ===")
    
    try:
        from question_evaluate.evaluate_platform_independent import evaluate_questions
        
        # Use existing test questions
        questions_file = "storage/generated_question/test_0.json"
        output_file = "storage/generated_question/test_training_evaluation.json"
        
        print("Evaluating test questions...")
        results = evaluate_questions(
            model_path="ai/llama3.2",
            questions_file=questions_file,
            output_file=output_file
        )
        
        stats = results.get('statistics', {})
        print(f"OK Evaluation completed:")
        print(f"  Total questions: {stats.get('total_questions', 0)}")
        print(f"  Accuracy: {stats.get('overall_accuracy', 0):.2%}")
        
        return True, results
        
    except Exception as e:
        print(f"ERROR Question evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_model_operations():
    """Test basic model operations"""
    print("\n=== Testing Model Operations ===")
    
    try:
        from llm_clients.config_manager import LLMConfigManager
        
        # Test LLM config
        config_manager = LLMConfigManager()
        generation_models = config_manager.get_generation_models()
        
        print(f"OK Available models: {list(generation_models.keys())}")
        
        # Test if we can create basic training directories
        storage_path = Path("storage/models/test_training")
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create a simple training metadata file
        metadata = {
            "test_run": True,
            "timestamp": time.time(),
            "model_path": "ai/llama3.2",
            "status": "testing"
        }
        
        metadata_file = storage_path / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"OK Test metadata saved: {metadata_file}")
        return True
        
    except Exception as e:
        print(f"ERROR Model operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_simple_training_pipeline():
    """Create a simple training pipeline demonstration"""
    print("\n=== Simple Training Pipeline Demo ===")
    
    # Step 1: Generate questions
    print("Step 1: Generate training questions")
    gen_success, questions = test_question_generation()
    
    if not gen_success:
        print("Skipping remaining steps due to generation failure")
        return False
        
    # Step 2: Evaluate questions  
    print("Step 2: Evaluate questions")
    eval_success, results = test_question_evaluation()
    
    if not eval_success:
        print("Evaluation failed, but continuing...")
        
    # Step 3: Create training data structure
    print("Step 3: Create training data structure")
    
    training_data = {
        "iteration": 1,
        "generated_questions": len(questions),
        "evaluation_results": results.get('statistics', {}),
        "questions_sample": questions[:5],  # Store first 5 questions as sample
        "timestamp": time.time(),
        "status": "demo_completed"
    }
    
    # Save training iteration data
    iteration_file = Path("storage/models/simple_training_demo.json")
    with open(iteration_file, 'w') as f:
        json.dump(training_data, f, indent=2)
        
    print(f"OK Training iteration data saved: {iteration_file}")
    print(f"OK Generated {training_data['generated_questions']} questions")
    
    if 'overall_accuracy' in training_data['evaluation_results']:
        accuracy = training_data['evaluation_results']['overall_accuracy']
        print(f"OK Evaluation accuracy: {accuracy:.2%}")
        
    return True


def main():
    """Main test function"""
    print("R-Zero Platform-Independent Training Test")
    print("=" * 50)
    
    # Check basic prerequisites
    print("Checking basic setup...")
    
    storage_path = Path("storage")
    storage_path.mkdir(exist_ok=True)
    (storage_path / "models").mkdir(exist_ok=True)
    (storage_path / "generated_question").mkdir(exist_ok=True)
    
    print("OK Storage directories created")
    
    # Test model operations
    if not test_model_operations():
        print("Basic model operations failed")
        return False
        
    # Run simple training pipeline demo
    success = create_simple_training_pipeline()
    
    if success:
        print("\nSUCCESS Simple training pipeline test completed successfully!")
        print("This demonstrates that the platform-independent training system is working.")
        print("\nTo run full training, use: python run_platform_independent_training.py")
    else:
        print("\nERROR Training pipeline test failed.")
        
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)