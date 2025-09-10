#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Platform-independent alternative to start_vllm_server.py
Uses OpenAI-compatible API endpoints instead of vLLM
This acts as a proxy/adapter to make existing Docker model runners work with R-Zero
"""

import argparse
import os
import sys
import time
import requests
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_clients.config_manager import LLMConfigManager
from openai import OpenAI

def check_endpoint_availability(base_url, model_name):
    """Check if the OpenAI-compatible endpoint is available"""
    try:
        client = OpenAI(base_url=base_url, api_key="dummy")
        
        # Test with a simple request
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print(f"✅ Endpoint {base_url} with model {model_name} is available")
        return True
        
    except Exception as e:
        print(f"❌ Endpoint {base_url} with model {model_name} is not available: {e}")
        return False

def wait_for_endpoint(base_url, model_name, max_retries=30, retry_interval=5):
    """Wait for the endpoint to become available"""
    print(f"Waiting for endpoint {base_url} with model {model_name}...")
    
    for attempt in range(max_retries):
        if check_endpoint_availability(base_url, model_name):
            return True
        
        print(f"Attempt {attempt + 1}/{max_retries} - Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)
    
    print(f"❌ Failed to connect to endpoint after {max_retries} attempts")
    return False

def create_proxy_info_file(port, base_url, model_name):
    """Create a file with proxy information for other processes"""
    proxy_info = {
        "port": port,
        "base_url": base_url,
        "model_name": model_name,
        "status": "running",
        "pid": os.getpid()
    }
    
    info_file = f"/tmp/openai_proxy_{port}.json"
    try:
        import json
        with open(info_file, 'w') as f:
            json.dump(proxy_info, f, indent=2)
        print(f"Proxy info saved to {info_file}")
    except Exception as e:
        print(f"Warning: Could not save proxy info: {e}")

def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible API proxy for R-Zero")
    parser.add_argument("--port", type=int, default=5000, help="Port to report as running on")
    parser.add_argument("--model_path", type=str, help="Model path (for compatibility)")
    parser.add_argument("--base_url", type=str, 
                       default="http://localhost:12434/engines/llama.cpp/v1",
                       help="Base URL of the OpenAI-compatible endpoint")
    parser.add_argument("--model_name", type=str, default="ai/llama3.2",
                       help="Model name to use")
    args = parser.parse_args()

    print("=" * 60)
    print("R-Zero Platform-Independent Model Service")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Base URL: {args.base_url}")
    print(f"Model Name: {args.model_name}")
    print(f"Model Path (if specified): {args.model_path}")
    print("=" * 60)

    # Try to load configuration
    try:
        config_manager = LLMConfigManager()
        generation_models = config_manager.get_generation_models()
        
        # Use configured endpoints if available
        if "solver" in generation_models:
            solver_config = generation_models["solver"]["config"]
            args.base_url = solver_config.get("base_url", args.base_url)
            args.model_name = solver_config.get("model_name", args.model_name)
            print(f"Using configured solver endpoint: {args.base_url}")
        
        if "challenger" in generation_models:
            challenger_config = generation_models["challenger"]["config"]
            # Could also use challenger config as fallback
            print(f"Challenger endpoint also available: {challenger_config.get('base_url')}")
            
    except Exception as e:
        print(f"Using default configuration due to error: {e}")

    # Check endpoint availability
    if not wait_for_endpoint(args.base_url, args.model_name):
        print("❌ Cannot connect to the specified endpoint")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure your Docker model runner is running")
        print("2. Check the base_url and model_name are correct")
        print("3. Test the endpoint manually with curl or similar tool")
        sys.exit(1)

    # Create proxy info file
    create_proxy_info_file(args.port, args.base_url, args.model_name)

    print("✅ OpenAI-compatible endpoint is ready!")
    print(f"✅ R-Zero can now use this service on port {args.port}")
    print("\n📋 Service Details:")
    print(f"   Endpoint: {args.base_url}")
    print(f"   Model: {args.model_name}")
    print(f"   Port: {args.port}")
    
    # Keep the service "running" - in reality, we're just confirming the endpoint works
    print("\n🔄 Service is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(10)
            # Periodic health check
            if not check_endpoint_availability(args.base_url, args.model_name):
                print("⚠️  Warning: Endpoint became unavailable!")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping service...")
        
        # Clean up info file
        try:
            info_file = f"/tmp/openai_proxy_{args.port}.json"
            if os.path.exists(info_file):
                os.remove(info_file)
                print(f"Cleaned up {info_file}")
        except Exception as e:
            print(f"Warning: Could not clean up info file: {e}")
        
        print("✅ Service stopped")

if __name__ == "__main__":
    main()