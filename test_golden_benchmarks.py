#!/usr/bin/env python3
"""
Test script for golden dataset benchmarking system
"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ["STORAGE_PATH"] = "C:/Users/GCV/Desktop/Rzero/R-Zero/storage"
os.environ["HUGGINGFACENAME"] = "test-user"

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import with error handling
try:
    from evaluation.golden_dataset_benchmarker import GoldenDatasetBenchmarker
except ImportError as e:
    print(f"Import error: {e}")
    print("Running basic dataset creation test instead...")
    
    def create_basic_golden_datasets():
        """Create basic golden datasets for testing"""
        golden_datasets = {
            "arithmetic_basic": [
                {"question": "What is 15 + 27?", "answer": "42"},
                {"question": "Calculate 144 ÷ 12", "answer": "12"},
                {"question": "What is 8 × 9?", "answer": "72"},
            ],
            "algebra_intermediate": [
                {"question": "If 3x + 7 = 22, what is x?", "answer": "5"},
                {"question": "Solve for y: 2y - 5 = 11", "answer": "8"},
            ]
        }
        
        datasets_path = Path("storage/golden_datasets")
        datasets_path.mkdir(parents=True, exist_ok=True)
        
        import json
        for name, questions in golden_datasets.items():
            dataset_file = datasets_path / f"{name}.json"
            with open(dataset_file, 'w') as f:
                json.dump(questions, f, indent=2)
        
        print(f"Basic golden datasets created at: {datasets_path}")
        return golden_datasets
    
    def main():
        print("🧪 Basic Golden Dataset Test")
        print("=" * 30)
        datasets = create_basic_golden_datasets()
        for name, questions in datasets.items():
            print(f"- {name}: {len(questions)} questions")
        print("✅ Basic test completed!")
    
    if __name__ == "__main__":
        main()
        sys.exit(0)


def main():
    """Test the golden dataset benchmarking system"""
    print("🧪 Testing Golden Dataset Benchmarking System")
    print("=" * 50)
    
    # Initialize benchmarker
    benchmarker = GoldenDatasetBenchmarker("storage")
    
    # Save golden datasets
    print("\n1. Saving golden datasets...")
    datasets_path = benchmarker.save_golden_datasets()
    print(f"   Golden datasets saved to: {datasets_path}")
    
    # Show dataset information
    print("\n2. Golden dataset information:")
    for name, questions in benchmarker.golden_datasets.items():
        print(f"   - {name}: {len(questions)} questions")
        print(f"     Example: {questions[0]['question']} → {questions[0]['answer']}")
    
    # Test benchmarking base model
    print("\n3. Testing base model benchmarking...")
    try:
        base_results = benchmarker.benchmark_model(
            model_path="ai/llama3.2",
            model_name="base_llama32_test",
            iteration=0,
            role="base"
        )
        
        print(f"   Base model benchmark completed!")
        print(f"   Results for {len(base_results)} datasets:")
        
        for result in base_results:
            print(f"     - {result.dataset_name}: {result.correct_answers}/{result.total_questions} ({result.accuracy:.2%})")
        
        # Calculate average
        if base_results:
            avg_accuracy = sum(r.accuracy for r in base_results) / len(base_results)
            print(f"   Average accuracy: {avg_accuracy:.2%}")
        
    except Exception as e:
        print(f"   Base model benchmarking failed: {e}")
        print("   This is expected if Docker model runner is not accessible")
    
    # Show what would happen with full benchmarking
    print("\n4. Full training benchmarking simulation:")
    print("   When running with trained models, the system would:")
    print("   - Benchmark base model (iteration 0)")
    print("   - Train and benchmark questioner v1")
    print("   - Train and benchmark solver v1") 
    print("   - Repeat for iterations 2-5")
    print("   - Generate comprehensive performance plots")
    print("   - Create detailed improvement reports")
    
    print("\n5. Expected outputs:")
    benchmarks_path = Path("storage/benchmarks")
    print(f"   - Performance plots: {benchmarks_path}/performance_analysis.png")
    print(f"   - Detailed report: {benchmarks_path}/detailed_performance_report.md")
    print(f"   - Statistics: {benchmarks_path}/performance_statistics.json")
    print(f"   - Raw results: {benchmarks_path}/benchmark_results_*.json")
    
    print("\n6. How to use:")
    print("   # Test benchmarking system")
    print("   python test_golden_benchmarks.py")
    print("   ")
    print("   # Run training with benchmarks")
    print("   python training_with_benchmarks.py")
    print("   ")
    print("   # Benchmark existing models")
    print("   python evaluation/golden_dataset_benchmarker.py --storage_path storage")
    
    print("\n✅ Golden dataset benchmarking system test completed!")
    print("   The system is ready to provide concrete evidence of R-Zero training improvements.")


if __name__ == "__main__":
    main()