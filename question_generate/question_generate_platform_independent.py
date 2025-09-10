"""
Platform-independent version of question_generate.py
Uses OpenAI-compatible API instead of vLLM for cross-platform compatibility
"""
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from evaluation.datasets_loader import get_dataset_handler
import json
import regex as re
import os
from openai import OpenAI
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_clients.config_manager import LLMConfigManager

STORAGE_PATH = os.getenv("STORAGE_PATH")

def extract_boxed(text):
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break   # no more \boxed{…}

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        if depth == 0:
            results.append(text[start + plen - 1: j])
        i = start + 1
    return results

def get_model_client():
    """Get model client using the configured generation model"""
    config_manager = LLMConfigManager()
    generation_models = config_manager.get_generation_models()
    
    if "challenger" in generation_models:
        model_config = generation_models["challenger"]
        
        if model_config["provider"] == "openai":
            client = OpenAI(
                base_url=model_config["config"].get("base_url"),
                api_key=model_config["config"].get("api_key", "dummy")
            )
            model_name = model_config["config"]["model_name"]
            return client, model_name
    
    # Fallback to Docker model runner
    client = OpenAI(
        base_url="http://localhost:12434/engines/llama.cpp/v1",
        api_key="dummy"
    )
    return client, "ai/llama3.2"

def generate_questions(client, model_name, prompts, max_tokens=2048, temperature=1.0, num_generations=1):
    """Generate responses using OpenAI-compatible API"""
    results = []
    
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_generations
            )
            
            # Extract content from response
            generated_texts = []
            for choice in response.choices:
                generated_texts.append(choice.message.content)
            
            results.append(generated_texts)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            results.append([f"Error: {e}"])
    
    return results

def generate_questions_simple(model_path, num_questions=10, output_file=None):
    """
    Simple wrapper function for generating questions
    Returns list of generated questions
    """
    print(f"Generating {num_questions} questions using model: {model_path}")
    
    # Get model client
    try:
        client, model_name = get_model_client()
        print(f"Connected to model: {model_name}")
    except Exception as e:
        print(f"Error connecting to model: {e}")
        return []

    # Load dataset for examples
    try:
        dataset_handler = get_dataset_handler("math")
        questions, answers = dataset_handler.load_data()
        example_question = questions[0]
        example_answer = answers[0]
    except Exception as e:
        print(f"Warning: Could not load dataset, using fallback: {e}")
        example_question = "What is 15 + 27?"
        example_answer = "42"
    
    # Prepare the generation prompt
    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "Generate a new, non-trivial mathematics problem. "
                "Output **exactly** the following format:\n\n"
                "<question>\n"
                "{The problem statement}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else."
            )
        },
        {"role": "user", "content": f"Example:\n\n{example_question}\n\nAnswer: {example_answer}\n\nNow generate a new problem:"},
    ]

    print(f"Generating {num_questions} questions...")
    
    # Generate questions
    prompts = [chat] * num_questions
    
    try:
        results = generate_questions(
            client, 
            model_name, 
            prompts,
            max_tokens=1024,
            temperature=1.0,
            num_generations=1
        )
        
        # Process results
        valid_questions = []
        for i, result_list in enumerate(results):
            for result in result_list:
                if not result.startswith("Error:"):
                    # Extract question and answer
                    question_match = re.search(r'<question>(.*?)</question>', result, re.DOTALL)
                    answer_boxes = extract_boxed(result)
                    
                    if question_match:
                        question_text = question_match.group(1).strip()
                        answer_text = answer_boxes[-1] if answer_boxes else "Unknown"
                        
                        valid_questions.append({
                            "question": question_text,
                            "answer": answer_text
                        })
                    else:
                        # Fallback - just use the result as question
                        valid_questions.append({
                            "question": result.strip(),
                            "answer": "Unknown"
                        })
        
        # Save to file if specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(valid_questions, f, indent=2)
            print(f"Saved {len(valid_questions)} questions to {output_file}")
        
        print(f"Generated {len(valid_questions)} questions")
        return valid_questions
        
    except Exception as e:
        print(f"Error during generation: {e}")
        # Return some fallback questions
        fallback_questions = [
            {"question": f"What is {i} + {i+1}?", "answer": str(2*i+1)}
            for i in range(1, num_questions + 1)
        ]
        return fallback_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--suffix", type=str, required=True)
    args = parser.parse_args()

    print(f"Using model: {args.model}")
    print(f"Suffix: {args.suffix}")

    # Get model client
    try:
        client, model_name = get_model_client()
        print(f"Connected to model: {model_name}")
    except Exception as e:
        print(f"Error connecting to model: {e}")
        return

    # Load dataset
    dataset_handler = get_dataset_handler("math")
    questions, answers = dataset_handler.load_data()
    question = questions[0]
    answer = answers[0]
    
    # Prepare the generation prompt
    chat = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                "Avoid re-using textbook clichés or famous contest problems.\n"
                "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                "<question>\n"
                "{The full problem statement on one or more lines}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else—no explanations, no extra markup."
            )
        },
        {"role": "user", "content": f"Here is an example:\n\n{question}\n\nAnswer: {answer}"},
    ]

    print("Generating questions...")
    
    # Generate multiple questions
    num_questions = 100  # Adjust as needed
    prompts = [chat] * num_questions
    
    try:
        results = generate_questions(
            client, 
            model_name, 
            prompts,
            max_tokens=2048,
            temperature=1.0,
            num_generations=1
        )
        
        # Process and save results
        valid_questions = []
        for i, result_list in enumerate(results):
            for result in result_list:
                if not result.startswith("Error:"):
                    # Extract question and answer
                    question_match = re.search(r'<question>(.*?)</question>', result, re.DOTALL)
                    answer_boxes = extract_boxed(result)
                    
                    if question_match and answer_boxes:
                        question_text = question_match.group(1).strip()
                        answer_text = answer_boxes[-1]  # Take the last boxed answer
                        
                        valid_questions.append({
                            "question": question_text,
                            "answer": answer_text,
                            "full_response": result
                        })
                        
                        if len(valid_questions) % 10 == 0:
                            print(f"Generated {len(valid_questions)} valid questions...")
        
        # Save results
        output_file = f"{STORAGE_PATH}/generated_question/generated_questions_{args.suffix}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(valid_questions, f, indent=2)
        
        print(f"Generated {len(valid_questions)} questions and saved to {output_file}")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()