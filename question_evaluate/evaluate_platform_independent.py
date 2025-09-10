#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Platform-independent version of evaluate.py
Uses OpenAI-compatible API instead of vLLM for cross-platform compatibility
'''

import json
from transformers import AutoTokenizer
import argparse
import re
import os
import sys
from pathlib import Path
from openai import OpenAI
try:
    import stopit  # Use the robust, thread-safe stopit library for timeouts
except ImportError:
    stopit = None
    print("Warning: stopit library not available. Install with: pip install stopit")

try:
    from mathruler.grader import extract_boxed_content, grade_answer
except ImportError:
    print("Warning: mathruler not available. Install with: pip install mathruler")
    def extract_boxed_content(text):
        return text
    def grade_answer(pred, gold):
        return pred.strip().lower() == gold.strip().lower()

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_clients.config_manager import LLMConfigManager

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate generated questions using OpenAI-compatible API.")
parser.add_argument("--model", type=str, default="ai/llama3.2", help="Model name to use for evaluation.")
parser.add_argument("--num_samples", type=int, default=9, help="Number of candidate answers to generate per question (n).")
parser.add_argument("--suffix", type=str, default="0", help="A unique suffix for file naming, often the GPU index.")
parser.add_argument("--save_name", type=str, required=True, help="A base name for input and output files.")
args = parser.parse_args()

# --- Constants and Paths ---
STORAGE_PATH = os.getenv("STORAGE_PATH")
if not STORAGE_PATH:
    print("Error: STORAGE_PATH environment variable not set")
    sys.exit(1)

INPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
OUTPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}_results.json"

# --- Timeout-Protected Grading Function ---
if stopit:
    @stopit.threading_timeoutable(default='TIMED_OUT')
    def grade_answer_with_timeout(res1, res2):
        """
        Wraps the mathruler 'grade_answer' function with a timeout.
        """
        return grade_answer(res1, res2)
else:
    def grade_answer_with_timeout(res1, res2, timeout=10):
        """
        Fallback without timeout protection
        """
        try:
            return grade_answer(res1, res2)
        except Exception as e:
            print(f"Grading error: {e}")
            return False

def get_model_client():
    """Get model client using the configured generation model"""
    try:
        config_manager = LLMConfigManager()
        generation_models = config_manager.get_generation_models()
        
        if "solver" in generation_models:
            model_config = generation_models["solver"]
            
            if model_config["provider"] == "openai":
                client = OpenAI(
                    base_url=model_config["config"].get("base_url"),
                    api_key=model_config["config"].get("api_key", "dummy")
                )
                model_name = model_config["config"]["model_name"]
                return client, model_name
    except Exception as e:
        print(f"Error loading config: {e}")
    
    # Fallback to Docker model runner
    client = OpenAI(
        base_url="http://localhost:12434/engines/llama.cpp/v1",
        api_key="dummy"
    )
    return client, args.model

def generate_answers(client, model_name, questions, num_samples=9):
    """Generate answers using OpenAI-compatible API"""
    results = []
    
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}")
        
        # Prepare the solving prompt
        messages = [
            {
                "role": "system", 
                "content": "You are an expert mathematician. Solve the problem step by step and provide your final answer in \\boxed{} format."
            },
            {
                "role": "user", 
                "content": f"Problem: {question}\n\nPlease solve this step by step and put your final answer in \\boxed{{}}."
            }
        ]
        
        question_results = []
        for sample in range(num_samples):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.8,  # Some randomness for diverse answers
                    top_p=0.9
                )
                
                answer = response.choices[0].message.content
                question_results.append(answer)
                
            except Exception as e:
                print(f"Error generating answer {sample+1} for question {i+1}: {e}")
                question_results.append(f"Error: {e}")
        
        results.append(question_results)
    
    return results

def extract_boxed(text):
    """Extract content from \\boxed{} format"""
    try:
        return extract_boxed_content(text)
    except:
        # Fallback extraction
        import re
        match = re.search(r'\\boxed\{([^}]*)\}', text)
        if match:
            return match.group(1)
        return text.strip()

def main():
    print(f"Loading questions from: {INPUT_FILE}")
    
    # Load questions
    try:
        with open(INPUT_FILE, 'r') as f:
            questions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found")
        sys.exit(1)
    
    # Extract questions and gold answers
    if isinstance(questions_data, list) and len(questions_data) > 0:
        if isinstance(questions_data[0], dict):
            questions = [item.get('question', item.get('problem', '')) for item in questions_data]
            gold_answers = [item.get('answer', item.get('solution', '')) for item in questions_data]
        else:
            questions = questions_data
            gold_answers = [''] * len(questions)  # No gold answers available
    else:
        print("Error: Invalid questions data format")
        sys.exit(1)
    
    print(f"Loaded {len(questions)} questions")
    
    # Get model client
    try:
        client, model_name = get_model_client()
        print(f"Connected to model: {model_name}")
    except Exception as e:
        print(f"Error connecting to model: {e}")
        sys.exit(1)
    
    # Generate answers
    print(f"Generating {args.num_samples} answers per question...")
    generated_answers = generate_answers(client, model_name, questions, args.num_samples)
    
    # Evaluate answers
    results = []
    for i, (question, gold_answer, answers) in enumerate(zip(questions, gold_answers, generated_answers)):
        print(f"Evaluating question {i+1}/{len(questions)}")
        
        question_result = {
            "question": question,
            "gold_answer": gold_answer,
            "generated_answers": answers,
            "evaluations": []
        }
        
        correct_count = 0
        for j, answer in enumerate(answers):
            if answer.startswith("Error:"):
                evaluation = {"answer": answer, "correct": False, "extracted": "", "error": True}
            else:
                # Extract boxed answer
                extracted = extract_boxed(answer)
                
                # Grade the answer
                if gold_answer:
                    if stopit:
                        is_correct = grade_answer_with_timeout(extracted, gold_answer, timeout=10)
                    else:
                        is_correct = grade_answer_with_timeout(extracted, gold_answer)
                    
                    if is_correct == 'TIMED_OUT':
                        is_correct = False
                        print(f"Grading timed out for question {i+1}, answer {j+1}")
                else:
                    is_correct = None  # No gold answer to compare against
                
                if is_correct:
                    correct_count += 1
                
                evaluation = {
                    "answer": answer,
                    "extracted": extracted,
                    "correct": is_correct,
                    "error": False
                }
            
            question_result["evaluations"].append(evaluation)
        
        question_result["correct_count"] = correct_count
        question_result["accuracy"] = correct_count / len(answers) if answers else 0
        
        results.append(question_result)
    
    # Calculate overall statistics
    total_questions = len(results)
    total_answers = sum(len(r["generated_answers"]) for r in results)
    total_correct = sum(r["correct_count"] for r in results)
    overall_accuracy = total_correct / total_answers if total_answers > 0 else 0
    
    final_results = {
        "model": model_name,
        "num_samples": args.num_samples,
        "statistics": {
            "total_questions": total_questions,
            "total_answers": total_answers,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy
        },
        "results": results
    }
    
    # Save results
    print(f"Saving results to: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Evaluation complete!")
    print(f"Overall accuracy: {overall_accuracy:.3f} ({total_correct}/{total_answers})")

if __name__ == "__main__":
    main()