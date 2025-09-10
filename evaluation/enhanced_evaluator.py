"""
Enhanced evaluation system for R-Zero training
Supports custom datasets and external challenger models
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_clients.config_manager import LLMConfigManager
from llm_clients.external_model_client import ChallengerModelManager
from llm_clients.langsmith_integration import LangSmithTracker
from question_evaluate.evaluate_platform_independent import get_model_client, extract_boxed


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    question: str
    gold_answer: str
    generated_answers: List[str]
    correct_count: int
    accuracy: float
    difficulty_score: float
    reasoning_quality: float


class EnhancedEvaluator:
    """Enhanced evaluation system with custom dataset support"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = LLMConfigManager(config_path)
        self.challenger_manager = ChallengerModelManager(self.config_manager)
        self.langsmith_tracker = LangSmithTracker(self.config_manager)
        self.storage_path = Path(os.getenv("STORAGE_PATH", "storage"))
        
    def load_custom_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load custom dataset from various formats"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        if dataset_path.suffix.lower() == '.json':
            return self._load_json_dataset(dataset_path)
        elif dataset_path.suffix.lower() == '.jsonl':
            return self._load_jsonl_dataset(dataset_path)
        elif dataset_path.suffix.lower() == '.csv':
            return self._load_csv_dataset(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    
    def _load_json_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON dataset"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'questions' in data:
                return data['questions']
            elif 'data' in data:
                return data['data']
            elif 'examples' in data:
                return data['examples']
        
        raise ValueError("Unable to parse JSON dataset structure")
    
    def _load_jsonl_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL dataset"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_csv_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load CSV dataset"""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            
            # Try to identify question and answer columns
            question_cols = [col for col in df.columns if 'question' in col.lower()]
            answer_cols = [col for col in df.columns if 'answer' in col.lower() or 'solution' in col.lower()]
            
            if question_cols and answer_cols:
                return [
                    {
                        'question': row[question_cols[0]],
                        'answer': row[answer_cols[0]]
                    }
                    for _, row in df.iterrows()
                ]
            else:
                # Assume first two columns are question and answer
                cols = list(df.columns)
                return [
                    {
                        'question': row[cols[0]],
                        'answer': row[cols[1]]
                    }
                    for _, row in df.iterrows()
                ]
        except ImportError:
            raise ImportError("pandas required for CSV support. Install with: pip install pandas")
    
    def normalize_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Normalize dataset to standard format"""
        normalized = []
        
        for item in dataset:
            # Try different field names for questions
            question = None
            for field in ['question', 'problem', 'query', 'input', 'text']:
                if field in item:
                    question = item[field]
                    break
            
            # Try different field names for answers
            answer = None
            for field in ['answer', 'solution', 'output', 'target', 'label']:
                if field in item:
                    answer = item[field]
                    break
            
            if question and answer:
                normalized.append({
                    'question': str(question).strip(),
                    'answer': str(answer).strip(),
                    'domain': item.get('domain', 'general'),
                    'difficulty': item.get('difficulty', 'medium'),
                    'metadata': {k: v for k, v in item.items() 
                               if k not in ['question', 'problem', 'query', 'input', 'text', 
                                          'answer', 'solution', 'output', 'target', 'label']}
                })
        
        return normalized
    
    def generate_questions_with_challenger(self, domain: str = "mathematics", 
                                         difficulty: str = "medium", 
                                         num_questions: int = 10) -> List[Dict[str, str]]:
        """Generate questions using the challenger model"""
        print(f"Generating {num_questions} {difficulty} questions in {domain}...")
        
        try:
            questions = self.challenger_manager.generate_challenging_questions(
                domain=domain,
                difficulty=difficulty,
                num_questions=num_questions
            )
            print(f"Generated {len(questions)} questions")
            
            # Log to LangSmith
            current_challenger = self.challenger_manager.get_current_challenger()
            if current_challenger and self.langsmith_tracker.is_available():
                challenger_name = self.challenger_manager.available_challengers.get(
                    self.challenger_manager.current_challenger, {}
                ).get("name", "unknown")
                
                self.langsmith_tracker.log_challenger_generation(
                    challenger_name, domain, difficulty, questions
                )
            
            return questions
        except Exception as e:
            print(f"Error generating questions: {e}")
            return self._fallback_questions(num_questions)
    
    def _fallback_questions(self, num_questions: int) -> List[Dict[str, str]]:
        """Generate fallback questions if challenger fails"""
        questions = []
        for i in range(num_questions):
            a, b = np.random.randint(2, 20, 2)
            op = np.random.choice(['+', '-', '*'])
            if op == '+':
                answer = a + b
            elif op == '-':
                answer = a - b
            else:
                answer = a * b
            
            questions.append({
                'question': f"Calculate {a} {op} {b}",
                'answer': str(answer)
            })
        
        return questions
    
    def evaluate_with_solver(self, questions: List[Dict[str, str]], 
                           num_samples: int = 5) -> Dict[str, Any]:
        """Evaluate questions using the solver model"""
        print(f"Evaluating {len(questions)} questions with {num_samples} samples each...")
        
        # Get solver client
        try:
            client, model_name = get_model_client()
            print(f"Using solver model: {model_name}")
        except Exception as e:
            print(f"Error connecting to solver: {e}")
            return {"error": str(e)}
        
        results = []
        total_correct = 0
        total_answers = 0
        
        for i, item in enumerate(questions):
            print(f"Evaluating question {i+1}/{len(questions)}")
            
            question = item['question']
            gold_answer = item['answer']
            
            # Generate multiple answers
            generated_answers = []
            correct_count = 0
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a mathematical problem solver. Solve the problem step by step and provide your final answer in \\boxed{} format."
                },
                {
                    "role": "user",
                    "content": f"Problem: {question}\n\nSolve step by step and put your final answer in \\boxed{{}}."
                }
            ]
            
            for j in range(num_samples):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.8
                    )
                    
                    answer = response.choices[0].message.content
                    generated_answers.append(answer)
                    
                    # Extract and evaluate answer
                    extracted = extract_boxed(answer)
                    if self._compare_answers(extracted, gold_answer):
                        correct_count += 1
                        total_correct += 1
                    
                    total_answers += 1
                    
                except Exception as e:
                    print(f"Error generating answer {j+1}: {e}")
                    generated_answers.append(f"Error: {e}")
                    total_answers += 1
            
            # Calculate metrics
            accuracy = correct_count / num_samples if num_samples > 0 else 0
            difficulty_score = self._calculate_difficulty_score(question, correct_count, num_samples)
            reasoning_quality = self._assess_reasoning_quality(generated_answers)
            
            result = EvaluationResult(
                question=question,
                gold_answer=gold_answer,
                generated_answers=generated_answers,
                correct_count=correct_count,
                accuracy=accuracy,
                difficulty_score=difficulty_score,
                reasoning_quality=reasoning_quality
            )
            
            results.append(result)
        
        # Calculate overall statistics
        overall_accuracy = total_correct / total_answers if total_answers > 0 else 0
        avg_difficulty = np.mean([r.difficulty_score for r in results])
        avg_reasoning = np.mean([r.reasoning_quality for r in results])
        
        return {
            "model": model_name,
            "statistics": {
                "total_questions": len(questions),
                "total_answers": total_answers,
                "total_correct": total_correct,
                "overall_accuracy": overall_accuracy,
                "average_difficulty_score": avg_difficulty,
                "average_reasoning_quality": avg_reasoning
            },
            "results": [
                {
                    "question": r.question,
                    "gold_answer": r.gold_answer,
                    "generated_answers": r.generated_answers,
                    "correct_count": r.correct_count,
                    "accuracy": r.accuracy,
                    "difficulty_score": r.difficulty_score,
                    "reasoning_quality": r.reasoning_quality
                }
                for r in results
            ]
        }
    
    def _compare_answers(self, predicted: str, gold: str) -> bool:
        """Compare predicted and gold answers"""
        try:
            # Try numerical comparison first
            pred_num = float(predicted.strip())
            gold_num = float(gold.strip())
            return abs(pred_num - gold_num) < 1e-6
        except (ValueError, TypeError):
            # Fall back to string comparison
            return predicted.strip().lower() == gold.strip().lower()
    
    def _calculate_difficulty_score(self, question: str, correct_count: int, total_samples: int) -> float:
        """Calculate difficulty score based on solver performance"""
        if total_samples == 0:
            return 0.5
        
        accuracy = correct_count / total_samples
        
        # Difficulty is inversely related to accuracy
        # Also consider question complexity factors
        complexity_factors = [
            len(question.split()) / 20.0,  # Question length
            question.count('(') + question.count('['),  # Parentheses
            len([w for w in question.split() if w.isdigit()]) / 10.0  # Number count
        ]
        
        complexity_score = min(1.0, sum(complexity_factors) / len(complexity_factors))
        difficulty = (1.0 - accuracy) * 0.7 + complexity_score * 0.3
        
        return max(0.0, min(1.0, difficulty))
    
    def _assess_reasoning_quality(self, answers: List[str]) -> float:
        """Assess the quality of reasoning in generated answers"""
        if not answers:
            return 0.0
        
        scores = []
        for answer in answers:
            if answer.startswith("Error:"):
                scores.append(0.0)
                continue
            
            # Simple heuristics for reasoning quality
            score = 0.0
            
            # Check for step-by-step reasoning
            if "step" in answer.lower() or "first" in answer.lower() or "then" in answer.lower():
                score += 0.3
            
            # Check for mathematical notation
            if "\\boxed{" in answer or "=" in answer:
                score += 0.2
            
            # Check for explanation
            if len(answer.split()) > 20:  # Longer answers likely have more explanation
                score += 0.2
            
            # Check for keywords indicating reasoning
            reasoning_words = ["because", "since", "therefore", "thus", "so", "hence"]
            if any(word in answer.lower() for word in reasoning_words):
                score += 0.3
            
            scores.append(min(1.0, score))
        
        return np.mean(scores)
    
    def run_comprehensive_evaluation(self, custom_dataset_path: Optional[str] = None,
                                   num_generated_questions: int = 10,
                                   num_samples_per_question: int = 5,
                                   domain: str = "mathematics",
                                   difficulty: str = "medium") -> Dict[str, Any]:
        """Run comprehensive evaluation with both custom and generated questions"""
        
        print("=== Starting Comprehensive Evaluation ===")
        
        # Start LangSmith experiment
        experiment_config = {
            "custom_dataset_path": custom_dataset_path,
            "num_generated_questions": num_generated_questions,
            "num_samples_per_question": num_samples_per_question,
            "domain": domain,
            "difficulty": difficulty
        }
        
        experiment_id = self.langsmith_tracker.start_experiment(
            f"comprehensive_evaluation_{int(time.time())}", 
            experiment_config
        )
        
        all_questions = []
        
        # Load custom dataset if provided
        if custom_dataset_path:
            print(f"Loading custom dataset: {custom_dataset_path}")
            try:
                raw_dataset = self.load_custom_dataset(custom_dataset_path)
                normalized_dataset = self.normalize_dataset(raw_dataset)
                all_questions.extend(normalized_dataset)
                print(f"Loaded {len(normalized_dataset)} questions from custom dataset")
            except Exception as e:
                print(f"Error loading custom dataset: {e}")
        
        # Generate additional questions with challenger
        if num_generated_questions > 0:
            generated_questions = self.generate_questions_with_challenger(
                domain=domain,
                difficulty=difficulty,
                num_questions=num_generated_questions
            )
            all_questions.extend(generated_questions)
            print(f"Generated {len(generated_questions)} additional questions")
        
        if not all_questions:
            raise ValueError("No questions available for evaluation")
        
        # Run evaluation
        results = self.evaluate_with_solver(all_questions, num_samples_per_question)
        
        # Log to LangSmith
        if self.langsmith_tracker.is_available():
            self.langsmith_tracker.log_training_iteration(
                1,  # Single evaluation iteration
                {"evaluation_type": "comprehensive"},
                all_questions,
                results
            )
        
        # Add metadata
        results["evaluation_config"] = {
            "custom_dataset_path": custom_dataset_path,
            "num_generated_questions": num_generated_questions,
            "num_samples_per_question": num_samples_per_question,
            "domain": domain,
            "difficulty": difficulty,
            "challenger_model": self.challenger_manager.current_challenger,
            "timestamp": time.time()
        }
        
        print("=== Evaluation Complete ===")
        print(f"Overall accuracy: {results['statistics']['overall_accuracy']:.3f}")
        print(f"Average difficulty: {results['statistics'].get('average_difficulty_score', 0):.3f}")
        print(f"Average reasoning quality: {results['statistics'].get('average_reasoning_quality', 0):.3f}")
        
        # End LangSmith experiment
        if self.langsmith_tracker.is_available():
            final_metrics = {
                "overall_accuracy": results['statistics']['overall_accuracy'],
                "average_difficulty_score": results['statistics'].get('average_difficulty_score', 0),
                "average_reasoning_quality": results['statistics'].get('average_reasoning_quality', 0),
                "total_questions": len(all_questions),
                "experiment_id": experiment_id
            }
            self.langsmith_tracker.end_experiment(final_metrics)
            
            # Print LangSmith URL if available
            experiment_url = self.langsmith_tracker.get_experiment_url()
            if experiment_url:
                print(f"🔗 View detailed results in LangSmith: {experiment_url}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results to file"""
        output_path = self.storage_path / "evaluation_results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def main():
    """Command line interface for enhanced evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced evaluation system")
    parser.add_argument("--dataset", type=str, help="Path to custom dataset")
    parser.add_argument("--generated", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per question")
    parser.add_argument("--domain", type=str, default="mathematics", help="Domain for generated questions")
    parser.add_argument("--difficulty", type=str, default="medium", help="Difficulty level")
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument("--challenger", type=str, help="Challenger model to use")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = EnhancedEvaluator()
    
    # Set challenger if specified
    if args.challenger:
        if not evaluator.challenger_manager.set_challenger(args.challenger):
            print("Available challengers:", evaluator.challenger_manager.list_challengers())
            return
    
    # Run evaluation
    try:
        results = evaluator.run_comprehensive_evaluation(
            custom_dataset_path=args.dataset,
            num_generated_questions=args.generated,
            num_samples_per_question=args.samples,
            domain=args.domain,
            difficulty=args.difficulty
        )
        
        # Save results
        if args.output:
            evaluator.save_results(results, args.output)
        else:
            timestamp = int(time.time())
            filename = f"enhanced_evaluation_{timestamp}.json"
            evaluator.save_results(results, filename)
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()