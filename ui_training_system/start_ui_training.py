#!/usr/bin/env python3
"""
R-Zero Training System with UI
Main entry point for starting the training system with web interface
"""

import os
import sys
import argparse
import asyncio
import subprocess
import time
from pathlib import Path
import webbrowser
import signal
import threading

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from backend.training.ui_training_manager import UITrainingManager, UITrainingConfig


def check_langsmith_setup():
    """Check if LangSmith is properly configured"""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("⚠️  Warning: LANGSMITH_API_KEY environment variable not set")
        print("   LangSmith integration will be disabled")
        print("   To enable LangSmith tracking:")
        print("   1. Get your API key from https://smith.langchain.com")
        print("   2. Set the environment variable: export LANGSMITH_API_KEY=your_key_here")
        print()
        return False
    else:
        print("✅ LangSmith API key found")
        return True


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'transformers', 
        'datasets', 'langsmith', 'websockets'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("✅ All required packages found")
    return True


def start_backend_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI backend server"""
    backend_path = Path(__file__).parent / "backend" / "api"
    
    print(f"🚀 Starting backend server on {host}:{port}")
    
    # Change to backend directory
    os.chdir(backend_path)
    
    # Start uvicorn server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app",
        "--host", host,
        "--port", str(port),
        "--reload"
    ]
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def start_frontend_dev_server():
    """Start the frontend development server"""
    frontend_path = Path(__file__).parent / "frontend"
    
    if not (frontend_path / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
    
    print("🎨 Starting frontend development server...")
    
    cmd = ["npm", "run", "dev"]
    return subprocess.Popen(cmd, cwd=frontend_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def wait_for_server(host: str, port: int, timeout: int = 30):
    """Wait for server to be ready"""
    import requests
    
    url = f"http://{host}:{port}/health"
    
    for i in range(timeout):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    
    return False


def run_training_example():
    """Run a simple training example to test the system"""
    config = UITrainingConfig(
        base_model="meta-llama/Llama-3.2-1B",
        experiment_name="test-ui-training",
        storage_path="./storage",
        num_iterations=2,
        questions_per_iteration=10,
        max_steps_questioner=3,
        max_steps_solver=5,
        enable_langsmith=os.getenv("LANGSMITH_API_KEY") is not None
    )
    
    print("🧪 Starting example training session...")
    
    async def run_example():
        manager = UITrainingManager(config)
        await manager.run_full_training_pipeline()
    
    asyncio.run(run_example())


def main():
    parser = argparse.ArgumentParser(description="R-Zero Training System with UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--backend-only", action="store_true", help="Start backend server only")
    parser.add_argument("--example", action="store_true", help="Run a training example")
    parser.add_argument("--check-setup", action="store_true", help="Check system setup")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 R-Zero Training System with UI")
    print("=" * 60)
    
    # Check setup
    if args.check_setup or not check_dependencies():
        check_langsmith_setup()
        if args.check_setup:
            return
        if not check_dependencies():
            return 1
    
    langsmith_configured = check_langsmith_setup()
    
    if args.example:
        run_training_example()
        return
    
    # Start backend server
    backend_process = start_backend_server(args.host, args.backend_port)
    
    processes = [backend_process]
    
    try:
        # Wait for backend to be ready
        print("⏳ Waiting for backend server to start...")
        if not wait_for_server(args.host, args.backend_port):
            print("❌ Backend server failed to start")
            return 1
        
        print(f"✅ Backend server started at http://{args.host}:{args.backend_port}")
        
        if not args.backend_only:
            # Start frontend server
            try:
                frontend_process = start_frontend_dev_server()
                processes.append(frontend_process)
                
                # Wait a bit for frontend to start
                time.sleep(3)
                
                frontend_url = f"http://localhost:{args.frontend_port}"
                print(f"✅ Frontend server started at {frontend_url}")
                
                if not args.no_browser:
                    print("🌐 Opening browser...")
                    webbrowser.open(frontend_url)
                
            except Exception as e:
                print(f"⚠️  Frontend server failed to start: {e}")
                print(f"Backend API available at http://{args.host}:{args.backend_port}")
        
        print("\n" + "=" * 60)
        print("🎉 R-Zero Training System is ready!")
        print("=" * 60)
        print(f"Backend API: http://{args.host}:{args.backend_port}")
        if not args.backend_only:
            print(f"Frontend UI:  http://localhost:{args.frontend_port}")
        print()
        print("Features:")
        print("✨ Web-based training interface")
        print("📊 Real-time training progress tracking")
        if langsmith_configured:
            print("📈 LangSmith experiment tracking enabled")
        else:
            print("📈 LangSmith experiment tracking disabled (set LANGSMITH_API_KEY to enable)")
        print("🔄 Live training metrics and visualization")
        print("📁 Model management and history")
        print()
        print("Press Ctrl+C to stop all servers")
        print("=" * 60)
        
        # Wait for interrupt
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    
    finally:
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        print("✅ Servers stopped")


if __name__ == "__main__":
    sys.exit(main())