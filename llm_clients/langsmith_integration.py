"""
LangSmith Integration for R-Zero Training System
Provides experiment tracking, evaluation logging, and performance monitoring
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

try:
    from langsmith import Client
    from langsmith.schemas import Run, Example
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("LangSmith not available. Install with: pip install langsmith")

from .config_manager import LLMConfigManager


class LangSmithTracker:
    """LangSmith integration for experiment tracking"""
    
    def __init__(self, config_manager: Optional[LLMConfigManager] = None):
        self.config_manager = config_manager or LLMConfigManager()
        self.client = None
        self.current_session = None
        self.enabled = False
        
        # Set the API key as environment variable for your token
        os.environ['LANGSMITH_API_KEY'] = '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'
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LangSmith client"""
        if not LANGSMITH_AVAILABLE:
            print("LangSmith not available, tracking disabled")
            return
        
        try:
            # Get LangSmith configuration
            config = self.config_manager.config.get('langsmith', {})
            
            # Use the token directly
            api_key = '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'
            project = config.get('project', 'rzero-training')
            endpoint = config.get('endpoint', 'https://api.smith.langchain.com')
            self.enabled = config.get('enabled', True)
            
            if not self.enabled:
                print("LangSmith tracking disabled in config")
                return
            
            # Initialize client
            self.client = Client(
                api_key=api_key,
                api_url=endpoint
            )
            
            # Set default project
            self.project_name = project
            
            # Test connection
            try:
                projects = list(self.client.list_projects(limit=1))
                print(f"✅ LangSmith connected successfully to project: {project}")
                self.enabled = True
            except Exception as e:
                print(f"⚠️ LangSmith connection test failed: {e}")
                self.enabled = False
            
        except Exception as e:
            print(f"Error initializing LangSmith: {e}")
            self.enabled = False
    
    def start_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """Start a new experiment session"""
        if not self.enabled or not self.client:
            return str(uuid.uuid4())
        
        try:
            # Create a new session
            session = self.client.create_session(
                session_name=f"{experiment_name}_{int(time.time())}",
                description=f"R-Zero training experiment: {experiment_name}",
                metadata={
                    "experiment_type": "rzero_training",
                    "config": config,
                    "start_time": time.time()
                }
            )
            
            self.current_session = session.id
            print(f"📊 Started LangSmith experiment: {experiment_name}")
            return str(session.id)
            
        except Exception as e:
            print(f"Error starting LangSmith experiment: {e}")
            return str(uuid.uuid4())
    
    def log_training_iteration(self, iteration: int, metrics: Dict[str, Any], 
                             questions: List[Dict], results: Dict[str, Any]):
        """Log a training iteration"""
        if not self.enabled or not self.client or not self.current_session:
            return
        
        try:
            # Log the training iteration as a run
            run_name = f"training_iteration_{iteration}"
            
            with self.client.trace(
                name=run_name,
                session_id=self.current_session,
                inputs={
                    "iteration": iteration,
                    "num_questions": len(questions),
                    "questions_sample": questions[:3] if questions else []
                },
                outputs={
                    "metrics": metrics,
                    "results_summary": {
                        "accuracy": results.get('statistics', {}).get('overall_accuracy', 0),
                        "total_questions": results.get('statistics', {}).get('total_questions', 0),
                        "total_correct": results.get('statistics', {}).get('total_correct', 0)
                    }
                },
                tags=["training", "iteration", f"iter_{iteration}"]
            ) as run:
                # Log individual question evaluations
                for i, (question_data, question_result) in enumerate(zip(questions, results.get('results', []))):
                    self._log_question_evaluation(
                        run.id,
                        question_data,
                        question_result,
                        f"question_{iteration}_{i}"
                    )
            
            print(f"📊 Logged training iteration {iteration} to LangSmith")
            
        except Exception as e:
            print(f"Error logging training iteration: {e}")
    
    def _log_question_evaluation(self, parent_run_id: str, question: Dict[str, Any], 
                                result: Dict[str, Any], run_name: str):
        """Log individual question evaluation"""
        try:
            with self.client.trace(
                name=run_name,
                parent_run_id=parent_run_id,
                inputs={
                    "question": question.get('question', ''),
                    "gold_answer": question.get('answer', ''),
                    "domain": question.get('domain', 'unknown'),
                    "difficulty": question.get('difficulty', 'medium')
                },
                outputs={
                    "generated_answers": result.get('generated_answers', []),
                    "correct_count": result.get('correct_count', 0),
                    "accuracy": result.get('accuracy', 0),
                    "difficulty_score": result.get('difficulty_score', 0),
                    "reasoning_quality": result.get('reasoning_quality', 0)
                },
                tags=["question", "evaluation"]
            ):
                pass
            
        except Exception as e:
            print(f"Error logging question evaluation: {e}")
    
    def log_challenger_generation(self, challenger_model: str, domain: str, 
                                difficulty: str, generated_questions: List[Dict[str, Any]]):
        """Log challenger question generation"""
        if not self.enabled or not self.client or not self.current_session:
            return
        
        try:
            with self.client.trace(
                name="challenger_generation",
                session_id=self.current_session,
                inputs={
                    "challenger_model": challenger_model,
                    "domain": domain,
                    "difficulty": difficulty,
                    "num_questions_requested": len(generated_questions)
                },
                outputs={
                    "questions_generated": len(generated_questions),
                    "questions": generated_questions,
                    "average_question_length": sum(len(q.get('question', '')) for q in generated_questions) / len(generated_questions) if generated_questions else 0
                },
                tags=["challenger", "generation", domain, difficulty]
            ):
                pass
            
            print(f"📊 Logged {len(generated_questions)} challenger questions to LangSmith")
            
        except Exception as e:
            print(f"Error logging challenger generation: {e}")
    
    def log_model_performance(self, model_name: str, performance_metrics: Dict[str, Any]):
        """Log model performance metrics"""
        if not self.enabled or not self.client or not self.current_session:
            return
        
        try:
            with self.client.trace(
                name="model_performance",
                session_id=self.current_session,
                inputs={
                    "model_name": model_name,
                    "evaluation_type": "comprehensive"
                },
                outputs=performance_metrics,
                tags=["performance", "evaluation", model_name]
            ):
                pass
            
            print(f"📊 Logged {model_name} performance to LangSmith")
            
        except Exception as e:
            print(f"Error logging model performance: {e}")
    
    def end_experiment(self, final_metrics: Dict[str, Any]):
        """End the current experiment"""
        if not self.enabled or not self.client or not self.current_session:
            return
        
        try:
            # Update session with final metrics
            self.client.update_session(
                session_id=self.current_session,
                end_time=time.time(),
                metadata={
                    "final_metrics": final_metrics,
                    "status": "completed"
                }
            )
            
            print(f"📊 Ended LangSmith experiment session")
            self.current_session = None
            
        except Exception as e:
            print(f"Error ending LangSmith experiment: {e}")
    
    def create_dataset(self, dataset_name: str, questions: List[Dict[str, Any]]) -> Optional[str]:
        """Create a dataset in LangSmith for evaluation"""
        if not self.enabled or not self.client:
            return None
        
        try:
            # Create dataset
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description=f"R-Zero training questions dataset - {len(questions)} questions"
            )
            
            # Add examples to dataset
            examples = []
            for i, question_data in enumerate(questions):
                example = self.client.create_example(
                    dataset_id=dataset.id,
                    inputs={
                        "question": question_data.get('question', ''),
                        "domain": question_data.get('domain', 'unknown'),
                        "difficulty": question_data.get('difficulty', 'medium')
                    },
                    outputs={
                        "answer": question_data.get('answer', ''),
                        "expected_reasoning": question_data.get('reasoning', '')
                    }
                )
                examples.append(example)
            
            print(f"📊 Created LangSmith dataset '{dataset_name}' with {len(examples)} examples")
            return str(dataset.id)
            
        except Exception as e:
            print(f"Error creating LangSmith dataset: {e}")
            return None
    
    def get_experiment_url(self) -> Optional[str]:
        """Get the URL to view the current experiment"""
        if not self.enabled or not self.current_session:
            return None
        
        return f"https://smith.langchain.com/projects/p/{self.project_name}/sessions/{self.current_session}"
    
    def is_available(self) -> bool:
        """Check if LangSmith integration is available and enabled"""
        return self.enabled and self.client is not None


def main():
    """Test LangSmith integration"""
    print("Testing LangSmith Integration")
    print("=" * 30)
    
    tracker = LangSmithTracker()
    
    if not tracker.is_available():
        print("❌ LangSmith not available")
        return
    
    # Test experiment
    experiment_id = tracker.start_experiment("test_experiment", {
        "challenger_model": "gpt-4o",
        "solver_model": "llama3.2",
        "num_iterations": 1
    })
    
    # Test question generation logging
    sample_questions = [
        {"question": "What is 2+2?", "answer": "4", "domain": "math", "difficulty": "easy"},
        {"question": "Calculate 15*7", "answer": "105", "domain": "math", "difficulty": "medium"}
    ]
    
    tracker.log_challenger_generation("gpt-4o", "mathematics", "easy", sample_questions)
    
    # Test training iteration logging
    sample_results = {
        "statistics": {
            "overall_accuracy": 0.85,
            "total_questions": 2,
            "total_correct": 1
        },
        "results": [
            {"correct_count": 1, "accuracy": 1.0, "difficulty_score": 0.2, "reasoning_quality": 0.8},
            {"correct_count": 0, "accuracy": 0.0, "difficulty_score": 0.7, "reasoning_quality": 0.6}
        ]
    }
    
    tracker.log_training_iteration(1, {"loss": 0.3, "learning_rate": 1e-5}, sample_questions, sample_results)
    
    # End experiment
    tracker.end_experiment({"final_accuracy": 0.85, "total_training_time": 120})
    
    if tracker.get_experiment_url():
        print(f"🔗 View experiment: {tracker.get_experiment_url()}")
    
    print("✅ LangSmith integration test completed")


if __name__ == "__main__":
    main()