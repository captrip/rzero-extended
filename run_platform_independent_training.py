#!/usr/bin/env python3
"""
Platform-Independent R-Zero Training Launcher
Simplified version that works with Docker model runner
"""

import os
import sys
import json
import time
from pathlib import Path

# Set environment variables for the training
os.environ["STORAGE_PATH"] = "C:/Users/GCV/Desktop/Rzero/R-Zero/storage"
os.environ["HUGGINGFACENAME"] = "test-user"

from training_platform_independent import TrainingConfig, PlatformIndependentTrainer


def run_simple_training():
    """Run simplified training that works with Docker model runner"""
    
    print("=== R-Zero Platform-Independent Training ===")
    print("Using Docker model runner at http://localhost:12434/engines/llama.cpp/v1")
    
    # Configuration for the training
    config = TrainingConfig(
        base_model="ai/llama3.2",  # This will use the Docker model runner
        experiment_name="llama32_platform_independent",
        storage_path="C:/Users/GCV/Desktop/Rzero/R-Zero/storage",
        huggingface_name="test-user",
        num_iterations=3,  # Start with 3 iterations to test
        questions_per_iteration=50,  # Smaller number for testing
        max_steps_questioner=3,  # Reduced for testing
        max_steps_solver=10,  # Reduced for testing
        learning_rate=5e-6,
        batch_size=2  # Smaller batch size for testing
    )
    
    print(f"Configuration:")
    print(f"  Base model: {config.base_model}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Questions per iteration: {config.questions_per_iteration}")
    print(f"  Storage path: {config.storage_path}")
    
    # Create trainer
    trainer = PlatformIndependentTrainer(config)
    
    # Run training pipeline
    try:
        trainer.run_full_training_pipeline()
        print("\n=== Training Completed Successfully! ===")
        
        # Show final results
        if trainer.training_history:
            final_iteration = trainer.training_history[-1]
            print(f"Final models saved:")
            print(f"  Questioner: {final_iteration['questioner_path']}")
            print(f"  Solver: {final_iteration['solver_path']}")
            
            total_time = sum(r.get("training_time_seconds", 0) for r in trainer.training_history)
            print(f"Total training time: {total_time:.1f} seconds")
            
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


def check_prerequisites():
    """Check if all prerequisites are available"""
    print("Checking prerequisites...")
    
    # Check if storage directory exists
    storage_path = Path("C:/Users/GCV/Desktop/Rzero/R-Zero/storage")
    if not storage_path.exists():
        print(f"Creating storage directory: {storage_path}")
        storage_path.mkdir(parents=True, exist_ok=True)
        
    # Check if models directory exists
    models_path = storage_path / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Check if we can import required modules
    try:
        import torch
        import transformers
        from llm_clients.config_manager import LLMConfigManager
        print("✓ Required modules available")
    except ImportError as e:
        print(f"✗ Missing required module: {e}")
        return False
        
    # Check if Docker model runner is accessible
    try:
        from llm_clients.config_manager import LLMConfigManager
        config_manager = LLMConfigManager()
        generation_models = config_manager.get_generation_models()
        print(f"✓ LLM config loaded: {list(generation_models.keys())}")
    except Exception as e:
        print(f"⚠ LLM config warning: {e}")
        
    print("Prerequisites check completed.")
    return True


if __name__ == "__main__":
    print("R-Zero Platform-Independent Training Launcher")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("Prerequisites check failed. Please install missing dependencies.")
        sys.exit(1)
        
    # Run training
    success = run_simple_training()
    
    if success:
        print("\n🎉 Training pipeline completed successfully!")
        print("Check the storage/models/ directory for your trained models.")
    else:
        print("\n❌ Training pipeline failed.")
        sys.exit(1)