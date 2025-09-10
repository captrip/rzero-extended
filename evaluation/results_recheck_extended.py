"""
Extended results recheck with support for multiple model providers
Supports OpenAI, Docker model runners, and Ollama
"""
import json
from mathruler.grader import extract_boxed_content, grade_answer
import random
import argparse
import os
import time
import sys
from pathlib import Path

# Add the project root to Python path to import our custom modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_clients.model_factory import ModelFactory
from llm_clients.config_manager import LLMConfigManager


def process_example_with_fallback(answer, response, config_manager):
    """
    Process an example using primary model with fallback support
    
    Args:
        answer: The predicted answer
        response: The ground truth answer  
        config_manager: LLM configuration manager
        
    Returns:
        str: "Yes" or "No" indicating if answer is correct
    """
    evaluation_models = config_manager.get_evaluation_models()
    timeouts = config_manager.get_timeouts()
    retry_settings = config_manager.get_retry_settings()
    
    # Prepare the messages
    messages = [
        {"role": "system", "content": "You are a math answer checker."},
        {"role": "user", "content": f"Hi, there is a answer: {answer}\n\n, and the ground truth answer is: {response}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
    ]
    
    # Try primary model first
    if "primary" in evaluation_models:
        try:
            primary_config = evaluation_models["primary"]
            client = ModelFactory.create_client(primary_config["provider"], primary_config["config"])
            
            if client.is_available():
                result = client.generate_response(
                    messages, 
                    temperature=0.1,
                    max_tokens=10
                )
                if not result.startswith("Error:"):
                    return result
                else:
                    print(f"Primary model failed: {result}")
            else:
                print("Primary model not available")
        except Exception as e:
            print(f"Primary model error: {e}")
    
    # Try fallback models
    fallback_models = evaluation_models.get("fallback", [])
    for fallback in fallback_models:
        try:
            client = ModelFactory.create_client(fallback["provider"], fallback["config"])
            
            if client.is_available():
                result = client.generate_response(
                    messages,
                    temperature=0.1, 
                    max_tokens=10
                )
                if not result.startswith("Error:"):
                    return result
                else:
                    print(f"Fallback model {fallback['provider']} failed: {result}")
            else:
                print(f"Fallback model {fallback['provider']} not available")
        except Exception as e:
            print(f"Fallback model {fallback['provider']} error: {e}")
            continue
    
    # If all models fail, return default
    print("All evaluation models failed, defaulting to 'No'")
    return "No"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--config", type=str, help="Path to LLM configuration file")
    args = parser.parse_args()
    
    STORAGE_PATH = os.getenv("STORAGE_PATH")
    if not STORAGE_PATH:
        print("ERROR: STORAGE_PATH environment variable not set")
        return
    
    # Initialize configuration manager
    config_manager = LLMConfigManager(args.config)
    
    print("=" * 60)
    print("R-Zero Extended Results Recheck")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Storage Path: {STORAGE_PATH}")
    
    # Show available evaluation models
    eval_models = config_manager.get_evaluation_models()
    print("\nEvaluation Models:")
    if "primary" in eval_models:
        primary = eval_models["primary"]
        print(f"  Primary: {primary['provider']} - {primary['config'].get('model_name', 'N/A')}")
    
    fallbacks = eval_models.get("fallback", [])
    if fallbacks:
        print("  Fallbacks:")
        for i, fallback in enumerate(fallbacks, 1):
            print(f"    {i}. {fallback['provider']} - {fallback['config'].get('model_name', 'N/A')}")
    print("=" * 60)
    
    new_results = []
    
    # Process each dataset
    for dataset in [
        "math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"
    ]:
        results_file = f'{STORAGE_PATH}/evaluation/{args.model_name.replace("/","_")}/results_{dataset}.json'
        
        if not os.path.exists(results_file):
            print(f"Skipping {dataset}: results file not found")
            continue
        
        print(f"\nProcessing {dataset}...")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        processed_count = 0
        corrected_count = 0
        
        for i in range(len(results) - 1):  # Skip last item (usually summary)
            if results[i]['score'] < 0.5:
                try:
                    gpt_check = process_example_with_fallback(
                        results[i]['answer'],
                        results[i]['response'], 
                        config_manager
                    )
                    
                    if "yes" in gpt_check.lower():
                        results[i]['score'] = 1
                        corrected_count += 1
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"  Processed {processed_count} items, corrected {corrected_count}")
                
                except Exception as e:
                    print(f"  Error processing item {i}: {e}")
                    continue
        
        # Calculate final score
        final_score = round(sum([result['score'] for result in results[:-1]]) / len(results[:-1]) * 100, 2)
        
        result_entry = {
            'model': args.model_name,
            'dataset': dataset,
            'score': final_score,
            'processed_items': processed_count,
            'corrected_items': corrected_count
        }
        
        new_results.append(result_entry)
        print(f"  Final score: {final_score}% (processed: {processed_count}, corrected: {corrected_count})")
        
        # Save to JSONL file
        with open('final_results_extended.jsonl', 'a', encoding='utf-8') as f:
            json.dump(result_entry, f, ensure_ascii=False)
            f.write('\n')
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in new_results:
        print(f"{result['dataset']:>12}: {result['score']:>6.2f}% "
              f"(processed: {result['processed_items']}, corrected: {result['corrected_items']})")
    
    if new_results:
        avg_score = sum(r['score'] for r in new_results) / len(new_results)
        print(f"{'Average':>12}: {avg_score:>6.2f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    main()