import axios, { AxiosResponse } from 'axios';
import { io, Socket } from 'socket.io-client';
import { TrainingStatus } from '../types/Training';

export interface TrainingConfigRequest {
  base_model: string;
  experiment_name: string;
  storage_path?: string;
  huggingface_name?: string;
  num_iterations?: number;
  questions_per_iteration?: number;
  max_steps_questioner?: number;
  max_steps_solver?: number;
  learning_rate?: number;
  batch_size?: number;
  save_steps?: number;
  enable_langsmith?: boolean;
  langsmith_project?: string;
}

export interface TrainingResponse {
  success: boolean;
  message: string;
  experiment_id?: string;
}

export class TrainingService {
  private baseUrl: string;
  private socket: Socket | null = null;
  private statusCallback: ((status: TrainingStatus) => void) | null = null;
  private logCallback: ((message: string) => void) | null = null;
  private errorCallback: ((error: string) => void) | null = null;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    
    // Configure axios defaults
    axios.defaults.timeout = 30000; // 30 second timeout
  }

  /**
   * Connect to WebSocket for real-time updates
   */
  connect(
    onStatus: (status: TrainingStatus) => void,
    onLog?: (message: string) => void,
    onError?: (error: string) => void
  ) {
    this.statusCallback = onStatus;
    this.logCallback = onLog;
    this.errorCallback = onError;

    // Connect to WebSocket using the standard WebSocket API
    const wsUrl = this.baseUrl.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws';
    
    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected to training service');
        // Send ping to keep connection alive
        const interval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
          } else {
            clearInterval(interval);
          }
        }, 30000); // Ping every 30 seconds
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'training_status' && message.data) {
            this.statusCallback?.(message.data);
          } else if (message.type === 'log' && message.message) {
            this.logCallback?.(message.message);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
          if (this.statusCallback) {
            this.connect(this.statusCallback, this.logCallback, this.errorCallback);
          }
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.errorCallback?.('WebSocket connection error');
      };

      // Store reference for cleanup
      (this as any).ws = ws;
      
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      this.errorCallback?.('Failed to connect to training service');
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnect() {
    const ws = (this as any).ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
    (this as any).ws = null;
    this.statusCallback = null;
    this.logCallback = null;
    this.errorCallback = null;
  }

  /**
   * Start a new training experiment
   */
  async startTraining(config: TrainingConfigRequest): Promise<TrainingResponse> {
    try {
      const response: AxiosResponse<TrainingResponse> = await axios.post(
        `${this.baseUrl}/training/start`,
        config
      );
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to start training';
      throw new Error(message);
    }
  }

  /**
   * Pause current training
   */
  async pauseTraining(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseUrl}/training/pause`);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to pause training';
      throw new Error(message);
    }
  }

  /**
   * Resume paused training
   */
  async resumeTraining(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseUrl}/training/resume`);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to resume training';
      throw new Error(message);
    }
  }

  /**
   * Stop current training
   */
  async stopTraining(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseUrl}/training/stop`);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to stop training';
      throw new Error(message);
    }
  }

  /**
   * Get current training status
   */
  async getStatus(): Promise<TrainingStatus> {
    try {
      const response = await axios.get(`${this.baseUrl}/training/status`);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to get status';
      throw new Error(message);
    }
  }

  /**
   * Get training history
   */
  async getHistory(): Promise<{ history: any[] }> {
    try {
      const response = await axios.get(`${this.baseUrl}/training/history`);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to get history';
      throw new Error(message);
    }
  }

  /**
   * List all experiments
   */
  async listExperiments(): Promise<{ experiments: any[] }> {
    try {
      const response = await axios.get(`${this.baseUrl}/experiments`);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to list experiments';
      throw new Error(message);
    }
  }

  /**
   * List all trained models
   */
  async listModels(): Promise<{ models: any[] }> {
    try {
      const response = await axios.get(`${this.baseUrl}/models`);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Failed to list models';
      throw new Error(message);
    }
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; timestamp: number }> {
    try {
      const response = await axios.get(`${this.baseUrl}/health`);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.detail || error.message || 'Health check failed';
      throw new Error(message);
    }
  }
}