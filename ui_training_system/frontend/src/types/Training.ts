export interface TrainingStatus {
  experiment_id: string | null;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  current_iteration: number;
  total_iterations: number;
  current_phase: string;
  progress_percentage: number;
  current_step: number;
  total_steps: number;
  loss: number | null;
  learning_rate: number | null;
  eta_seconds: number | null;
  error_message: string | null;
  model_paths: Record<string, string> | null;
}

export interface TrainingConfig {
  base_model: string;
  experiment_name: string;
  storage_path: string;
  huggingface_name: string;
  num_iterations: number;
  questions_per_iteration: number;
  max_steps_questioner: number;
  max_steps_solver: number;
  learning_rate: number;
  batch_size: number;
  save_steps: number;
  enable_langsmith: boolean;
  langsmith_project: string;
}

export interface TrainingMetrics {
  iteration: number;
  role: 'questioner' | 'solver';
  loss: number;
  learning_rate: number;
  step: number;
  timestamp: number;
  model_path?: string;
  questions_generated?: number;
  evaluation_score?: number;
  training_time_seconds?: number;
}

export interface ExperimentInfo {
  experiment_id: string;
  experiment_name: string;
  status: string;
  start_time: number;
  config: TrainingConfig;
  final_models?: {
    questioner?: string;
    solver?: string;
  };
  total_training_time?: number;
  completed_iterations?: number;
}

export interface ModelInfo {
  name: string;
  path: string;
  metadata: {
    iteration?: number;
    role?: string;
    base_model?: string;
    training_steps?: number;
    timestamp?: number;
    questions_generated?: number;
    evaluation_results?: any;
  };
}

export interface TrainingHistory {
  iteration: number;
  questioner_path: string;
  solver_path: string;
  training_time_seconds: number;
  timestamp: number;
}