export interface User {
  id: number;
  email: string;
  name: string;
  created_at: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface Recording {
  id: number;
  wake_word: string;
  filename: string;
  duration_s: number;
  created_at: string;
}

export interface UploadResponse {
  recording_id: number;
  filename: string;
  duration_s: number;
}

export interface TrainingJob {
  job_id: number;
  status: TrainingStatus;
  progress: number;
  d_prime: number | null;
  model_id: number | null;
  error: string | null;
}

export type TrainingStatus =
  | "queued"
  | "training"
  | "completed"
  | "failed";

export interface TrainingStartResponse {
  job_id: number;
  status: TrainingStatus;
}

export interface TrainingEvent {
  status: TrainingStatus;
  progress: number;
  epoch: number;
  total_epochs: number;
  train_loss: number | null;
  val_loss: number | null;
  d_prime: number | null;
  model_id: number | null;
  message: string;
  error: string | null;
}

export interface Model {
  id: number;
  wake_word: string;
  d_prime: number | null;
  created_at: string;
  size_bytes: number;
}

export interface ModelConfig {
  d_prime: number | null;
  far_per_hour: number | null;
  frr: number | null;
  training_config: Record<string, unknown>;
}

export interface ModelPerformanceResponse {
  model_name: string;
  cohen_d: number | null;
  threshold: number | null;
  file_size: number;
  created_at: string;
  positive_scores: number[];
  negative_scores: number[];
  evaluation_available: boolean;
  evaluation_data: Record<string, unknown>;
}

export interface UsageResponse {
  models_used: number;
  models_limit: number | null;
  period_start: string;
  period_end: string;
}

export interface SubscriptionResponse {
  tier: string;
  status: string;
  current_period_end: string;
  usage: UsageResponse;
}

export interface BillingPortalResponse {
  url: string;
}
