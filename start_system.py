#!/usr/bin/env python
"""
System startup script for R-Zero Enhanced Training System
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Setup environment variables"""
    print("Setting up environment...")
    
    # Set LangSmith API token
    os.environ['LANGSMITH_API_KEY'] = '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'
    os.environ['LANGSMITH_PROJECT'] = 'rzero-training'
    os.environ['STORAGE_PATH'] = 'storage'
    
    print("Environment configured:")
    print(f"  LangSmith API Key: {os.environ['LANGSMITH_API_KEY'][:20]}...")
    print(f"  Storage Path: {os.environ['STORAGE_PATH']}")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "storage",
        "storage/models", 
        "storage/generated_question",
        "storage/evaluation_results",
        "storage/test_datasets"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")
    
    return True

def test_dependencies():
    """Test if required dependencies are available"""
    print("Testing dependencies...")
    
    required_modules = [
        'fastapi',
        'uvicorn', 
        'openai',
        'yaml',
        'numpy'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  {module}: OK")
        except ImportError:
            print(f"  {module}: MISSING")
            missing.append(module)
    
    # Test optional LangSmith
    try:
        import langsmith
        print("  langsmith: OK")
    except ImportError:
        print("  langsmith: MISSING (optional)")
    
    if missing:
        print(f"Warning: Missing modules: {missing}")
        print("Install with: pip install " + " ".join(missing))
    
    return len(missing) == 0

def start_api_server():
    """Start the API server"""
    print("Starting API server...")
    
    try:
        # Change to the UI directory
        ui_dir = Path("ui_training_system")
        api_file = ui_dir / "real_training_api.py"
        
        if not api_file.exists():
            print(f"Error: API file not found at {api_file}")
            return False
        
        # Start the server in background
        cmd = [sys.executable, str(api_file), "--host", "127.0.0.1", "--port", "8001"]
        
        print(f"Running: {' '.join(cmd)}")
        print("API server starting on http://127.0.0.1:8001")
        print("API docs available at: http://127.0.0.1:8001/docs")
        
        # Start server
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Give it a moment to start
        time.sleep(3)
        
        # Check if it's still running
        if process.poll() is None:
            print("API server started successfully!")
            return process
        else:
            print("API server failed to start")
            return False
            
    except Exception as e:
        print(f"Error starting API server: {e}")
        return False

def open_ui():
    """Open the UI in browser"""
    print("Opening UI...")
    
    ui_file = Path("ui_training_system/real_training_ui.html")
    
    if not ui_file.exists():
        print(f"Error: UI file not found at {ui_file}")
        return False
    
    try:
        import webbrowser
        ui_path = ui_file.resolve()
        webbrowser.open(f"file:///{ui_path}")
        print(f"UI opened: {ui_path}")
        return True
    except Exception as e:
        print(f"Could not open browser: {e}")
        print(f"Manually open: {ui_file.resolve()}")
        return True

def main():
    """Main startup sequence"""
    print("R-Zero Enhanced Training System Startup")
    print("=" * 50)
    
    # Setup
    if not setup_environment():
        print("Environment setup failed")
        return
    
    if not create_directories():
        print("Directory creation failed")
        return
    
    if not test_dependencies():
        print("Some dependencies missing, but continuing...")
    
    # Start API
    api_process = start_api_server()
    if not api_process:
        print("Failed to start API server")
        return
    
    # Open UI
    if not open_ui():
        print("Failed to open UI")
    
    print("\n" + "=" * 50)
    print("System is ready!")
    print("=" * 50)
    print("API Server: http://127.0.0.1:8001")
    print("API Docs: http://127.0.0.1:8001/docs")
    print(f"UI: file:///{Path('ui_training_system/real_training_ui.html').resolve()}")
    print("LangSmith: https://smith.langchain.com/")
    print("\nPress Ctrl+C to stop the API server")
    
    try:
        # Keep the script running and show API output
        while True:
            output = api_process.stdout.readline()
            if output:
                print(f"API: {output.strip()}")
            elif api_process.poll() is not None:
                print("API server stopped")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping API server...")
        api_process.terminate()
        print("System stopped")

if __name__ == "__main__":
    main()