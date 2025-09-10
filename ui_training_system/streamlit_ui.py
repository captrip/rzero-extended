#!/usr/bin/env python3
"""
Streamlit UI for R-Zero Training with LangSmith Integration
Real-time training monitoring and control interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import threading
import queue

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui_training_system.langsmith_tracker import LangSmithTracker, setup_langsmith_environment
from ui_training_system.ui_training_manager import UITrainingManager


# Streamlit page config
st.set_page_config(
    page_title="R-Zero Training Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}

.success-card {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
}

.warning-card {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
}

.error-card {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #dc3545;
}

.stProgress .st-bo {
    background-color: #ff6b6b;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'training_manager' not in st.session_state:
        st.session_state.training_manager = None
    if 'training_running' not in st.session_state:
        st.session_state.training_running = False
    if 'training_logs' not in st.session_state:
        st.session_state.training_logs = []
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = {}
    if 'current_iteration' not in st.session_state:
        st.session_state.current_iteration = 0
    if 'current_phase' not in st.session_state:
        st.session_state.current_phase = "idle"


def render_sidebar():
    """Render sidebar with training controls"""
    st.sidebar.header("🧠 R-Zero Training")
    
    # LangSmith setup check
    if not setup_langsmith_environment():
        st.sidebar.error("⚠️ LangSmith not configured")
        with st.sidebar.expander("Setup LangSmith"):
            st.code("""
# Set these environment variables:
export LANGCHAIN_API_KEY='your-api-key'
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT='r-zero-training'
            """)
    else:
        st.sidebar.success("✅ LangSmith configured")
    
    st.sidebar.markdown("---")
    
    # Training configuration
    st.sidebar.subheader("Training Configuration")
    
    experiment_name = st.sidebar.text_input(
        "Experiment Name", 
        value=f"rzero-{datetime.now().strftime('%m%d-%H%M')}"
    )
    
    base_model = st.sidebar.selectbox(
        "Base Model",
        ["ai/llama3.2", "custom_model_path"],
        index=0
    )
    
    num_iterations = st.sidebar.slider(
        "Training Iterations",
        min_value=1,
        max_value=10,
        value=3
    )
    
    questions_per_iter = st.sidebar.slider(
        "Questions per Iteration",
        min_value=10,
        max_value=500,
        value=50
    )
    
    batch_size = st.sidebar.selectbox(
        "Batch Size",
        [1, 2, 4, 8],
        index=1
    )
    
    learning_rate = st.sidebar.selectbox(
        "Learning Rate",
        [1e-6, 5e-6, 1e-5, 5e-5],
        index=1,
        format_func=lambda x: f"{x:.0e}"
    )
    
    enable_benchmarking = st.sidebar.checkbox(
        "Enable Golden Dataset Benchmarking",
        value=True
    )
    
    st.sidebar.markdown("---")
    
    # Training controls
    st.sidebar.subheader("Training Controls")
    
    config = {
        "experiment_name": experiment_name,
        "base_model": base_model,
        "num_iterations": num_iterations,
        "questions_per_iteration": questions_per_iter,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "enable_benchmarking": enable_benchmarking
    }
    
    if not st.session_state.training_running:
        if st.sidebar.button("🚀 Start Training", type="primary"):
            start_training(config)
    else:
        if st.sidebar.button("⏹️ Stop Training", type="secondary"):
            stop_training()
        
        st.sidebar.info(f"Training in progress...\nIteration: {st.session_state.current_iteration}\nPhase: {st.session_state.current_phase}")
    
    st.sidebar.markdown("---")
    
    # Model management
    st.sidebar.subheader("Model Management")
    
    if st.sidebar.button("📁 View Saved Models"):
        show_saved_models()
    
    if st.sidebar.button("📊 Load Previous Results"):
        load_previous_results()
    
    return config


def start_training(config):
    """Start training with given configuration"""
    try:
        # Initialize training manager
        st.session_state.training_manager = UITrainingManager(config)
        st.session_state.training_running = True
        st.session_state.current_iteration = 0
        st.session_state.current_phase = "initializing"
        
        # Clear previous data
        st.session_state.training_logs = []
        st.session_state.metrics_history = []
        st.session_state.benchmark_results = {}
        
        st.success(f"🚀 Training started: {config['experiment_name']}")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to start training: {e}")


def stop_training():
    """Stop current training"""
    st.session_state.training_running = False
    st.session_state.current_phase = "stopped"
    if st.session_state.training_manager:
        st.session_state.training_manager.stop_training()
    st.warning("⏹️ Training stopped")
    st.rerun()


def render_main_dashboard():
    """Render main dashboard content"""
    st.title("🧠 R-Zero Training Dashboard")
    st.markdown("Real-time monitoring of self-evolving reasoning model training")
    
    # Status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🟢 Running" if st.session_state.training_running else "🔴 Idle"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Status</h4>
            <p>{status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        iteration = st.session_state.current_iteration
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Iteration</h4>
            <p>{iteration}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        phase = st.session_state.current_phase.title()
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Phase</h4>
            <p>{phase}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_logs = len(st.session_state.training_logs)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Events Logged</h4>
            <p>{total_logs}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress tracking
    if st.session_state.training_running and st.session_state.training_manager:
        total_iterations = st.session_state.training_manager.config.get("num_iterations", 1)
        progress = st.session_state.current_iteration / total_iterations
        st.progress(progress, text=f"Training Progress: {st.session_state.current_iteration}/{total_iterations}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Metrics", "📊 Benchmarks", "📝 Logs", "🔍 Models", "⚙️ Settings"
    ])
    
    with tab1:
        render_metrics_tab()
    
    with tab2:
        render_benchmarks_tab()
    
    with tab3:
        render_logs_tab()
    
    with tab4:
        render_models_tab()
    
    with tab5:
        render_settings_tab()


def render_metrics_tab():
    """Render training metrics visualization"""
    st.subheader("📈 Training Metrics")
    
    if not st.session_state.metrics_history:
        st.info("No metrics data available. Start training to see metrics.")
        return
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(st.session_state.metrics_history)
    
    if len(metrics_df) == 0:
        st.info("No metrics collected yet.")
        return
    
    # Metrics plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy over iterations
        if 'accuracy' in metrics_df.columns:
            fig = px.line(
                metrics_df, 
                x='iteration', 
                y='accuracy',
                color='phase',
                title="Accuracy by Iteration",
                markers=True
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Loss over iterations
        if 'loss' in metrics_df.columns:
            fig = px.line(
                metrics_df,
                x='iteration',
                y='loss', 
                color='phase',
                title="Loss by Iteration",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    if len(metrics_df) > 0:
        st.dataframe(metrics_df, use_container_width=True)


def render_benchmarks_tab():
    """Render benchmark results"""
    st.subheader("📊 Golden Dataset Benchmarks")
    
    if not st.session_state.benchmark_results:
        st.info("No benchmark results available. Enable benchmarking in training configuration.")
        return
    
    # Benchmark results summary
    benchmark_data = []
    for iteration, phases in st.session_state.benchmark_results.items():
        for phase, datasets in phases.items():
            for dataset, results in datasets.items():
                if isinstance(results, dict) and 'accuracy' in results:
                    benchmark_data.append({
                        'iteration': iteration,
                        'phase': phase,
                        'dataset': dataset,
                        'accuracy': results['accuracy'],
                        'correct': results.get('correct_answers', 0),
                        'total': results.get('total_questions', 0)
                    })
    
    if not benchmark_data:
        st.info("No benchmark data collected yet.")
        return
    
    benchmark_df = pd.DataFrame(benchmark_data)
    
    # Benchmark visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy by dataset
        fig = px.bar(
            benchmark_df,
            x='dataset',
            y='accuracy',
            color='iteration',
            title="Accuracy by Dataset",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Progress over iterations
        fig = px.line(
            benchmark_df,
            x='iteration',
            y='accuracy',
            color='dataset',
            title="Benchmark Progress",
            markers=True
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    # Benchmark table
    st.subheader("Benchmark Results Table")
    st.dataframe(benchmark_df, use_container_width=True)


def render_logs_tab():
    """Render training logs"""
    st.subheader("📝 Training Logs")
    
    # Log level filter
    log_levels = st.multiselect(
        "Filter by Event Type",
        ["experiment_start", "iteration_start", "training_metrics", "benchmark_results", "model_saved", "error"],
        default=["iteration_start", "training_metrics", "benchmark_results", "error"]
    )
    
    # Display logs
    filtered_logs = [
        log for log in st.session_state.training_logs 
        if log.get('event', '') in log_levels
    ]
    
    if not filtered_logs:
        st.info("No logs available with current filters.")
        return
    
    # Recent logs (last 20)
    recent_logs = filtered_logs[-20:]
    
    for log in reversed(recent_logs):  # Show most recent first
        timestamp = datetime.fromtimestamp(log.get('timestamp', time.time())).strftime("%H:%M:%S")
        event = log.get('event', 'unknown')
        
        if event == 'error':
            st.markdown(f"""
            <div class="error-card">
                <strong>[{timestamp}] ERROR</strong><br>
                Iteration {log.get('iteration', 'N/A')} - {log.get('phase', 'N/A')}<br>
                {log.get('error', 'Unknown error')}
            </div>
            """, unsafe_allow_html=True)
            
        elif event == 'training_metrics':
            metrics = log.get('metrics', {})
            metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items() if isinstance(v, (int, float))])
            st.markdown(f"""
            <div class="success-card">
                <strong>[{timestamp}] METRICS</strong><br>
                Iteration {log.get('iteration', 'N/A')} - {log.get('phase', 'N/A')}<br>
                {metrics_str}
            </div>
            """, unsafe_allow_html=True)
            
        elif event == 'benchmark_results':
            st.markdown(f"""
            <div class="warning-card">
                <strong>[{timestamp}] BENCHMARK</strong><br>
                Iteration {log.get('iteration', 'N/A')} - {log.get('phase', 'N/A')}<br>
                Benchmark evaluation completed
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="metric-card">
                <strong>[{timestamp}] {event.upper()}</strong><br>
                Iteration {log.get('iteration', 'N/A')} - {log.get('phase', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)


def render_models_tab():
    """Render saved models information"""
    st.subheader("🔍 Saved Models")
    
    models_path = Path("storage/models")
    if not models_path.exists():
        st.info("No models directory found.")
        return
    
    # Scan for models
    saved_models = []
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            hf_path = model_dir / "huggingface"
            metadata_path = model_dir / "training_metadata.json"
            
            model_info = {
                "name": model_dir.name,
                "path": str(model_dir),
                "has_hf_model": hf_path.exists(),
                "has_metadata": metadata_path.exists(),
                "size": "N/A"
            }
            
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    model_info.update({
                        "iteration": metadata.get("iteration", "N/A"),
                        "role": metadata.get("role", "N/A"),
                        "timestamp": metadata.get("timestamp", "N/A")
                    })
                except:
                    pass
            
            saved_models.append(model_info)
    
    if not saved_models:
        st.info("No saved models found.")
        return
    
    # Models table
    models_df = pd.DataFrame(saved_models)
    st.dataframe(models_df, use_container_width=True)
    
    # Model actions
    st.subheader("Model Actions")
    selected_model = st.selectbox(
        "Select Model",
        options=[m["name"] for m in saved_models],
        index=0 if saved_models else None
    )
    
    if selected_model:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Benchmark Selected Model"):
                st.info(f"Benchmarking {selected_model}...")
                # TODO: Add benchmarking functionality
        
        with col2:
            if st.button("📁 Export Model"):
                st.info(f"Exporting {selected_model}...")
                # TODO: Add export functionality
        
        with col3:
            if st.button("🗑️ Delete Model"):
                st.warning(f"Delete {selected_model}? (Not implemented)")


def render_settings_tab():
    """Render settings and configuration"""
    st.subheader("⚙️ Settings")
    
    # LangSmith configuration
    st.subheader("LangSmith Configuration")
    
    current_api_key = os.getenv("LANGCHAIN_API_KEY", "")
    api_key_masked = f"{'*' * (len(current_api_key) - 8)}{current_api_key[-8:]}" if len(current_api_key) > 8 else "Not set"
    
    st.info(f"Current API Key: {api_key_masked}")
    
    if st.button("🔄 Refresh LangSmith Connection"):
        if setup_langsmith_environment():
            st.success("✅ LangSmith connection verified")
        else:
            st.error("❌ LangSmith connection failed")
    
    # Storage settings
    st.subheader("Storage Settings")
    
    storage_path = Path("storage")
    if storage_path.exists():
        # Calculate storage usage
        total_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        st.info(f"Storage Usage: {size_mb:.1f} MB")
        
        if st.button("🧹 Clean Old Logs"):
            st.info("Cleaning old logs... (Not implemented)")
    
    # Export settings
    st.subheader("Data Export")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Training Logs"):
            # Export logs as JSON
            logs_json = json.dumps(st.session_state.training_logs, indent=2, default=str)
            st.download_button(
                label="Download Logs JSON",
                data=logs_json,
                file_name=f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📥 Export Metrics Data"):
            # Export metrics as CSV
            if st.session_state.metrics_history:
                metrics_df = pd.DataFrame(st.session_state.metrics_history)
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Download Metrics CSV",
                    data=csv,
                    file_name=f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def load_previous_results():
    """Load previous training results"""
    st.info("Loading previous results... (Not implemented)")
    # TODO: Implement loading from LangSmith logs


def show_saved_models():
    """Show saved models in sidebar"""
    st.info("Viewing saved models in Models tab")


def simulate_training_update():
    """Simulate training updates for demo"""
    if st.session_state.training_running and st.session_state.training_manager:
        # Simulate progress
        st.session_state.current_iteration += 0.1
        
        # Add fake metrics
        fake_metrics = {
            "timestamp": time.time(),
            "iteration": int(st.session_state.current_iteration),
            "phase": st.session_state.current_phase,
            "accuracy": 0.45 + (st.session_state.current_iteration * 0.05),
            "loss": 2.0 - (st.session_state.current_iteration * 0.1)
        }
        
        st.session_state.metrics_history.append(fake_metrics)
        
        # Add fake log
        fake_log = {
            "event": "training_metrics",
            "timestamp": time.time(),
            "iteration": int(st.session_state.current_iteration),
            "phase": st.session_state.current_phase,
            "metrics": {"accuracy": fake_metrics["accuracy"], "loss": fake_metrics["loss"]}
        }
        
        st.session_state.training_logs.append(fake_log)


def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Auto-refresh every 5 seconds when training is running
    if st.session_state.training_running:
        time.sleep(1)  # Small delay
        simulate_training_update()  # For demo purposes
        st.rerun()
    
    # Render UI
    config = render_sidebar()
    render_main_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("🧠 **R-Zero Training Dashboard** | Powered by Streamlit & LangSmith")


if __name__ == "__main__":
    main()