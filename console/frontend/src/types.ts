export interface User {
  id: number;
  email: string;
  name: string;
  email_verified: boolean;
  created_at?: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface MessageResponse {
  message: string;
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
  wake_word: string;
  duration_s: number;
}

export interface BulkUploadItem {
  filename: string;
  status: "success" | "error";
  recording_id: number | null;
  wake_word: string | null;
  duration_s: number | null;
  error: string | null;
}

export interface BulkUploadResponse {
  results: BulkUploadItem[];
  uploaded: number;
  failed: number;
}

export interface RecordingCountResponse {
  wake_word: string | null;
  count: number;
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
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface TrainingStartResponse {
  job_id: number;
  status: TrainingStatus;
}

export interface CircuitBreakerState {
  consecutive_failures: number;
  paused: boolean;
  next_attempt_at: string | null;
  last_failure_at: string | null;
  pause_reason: string | null;
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

export type QualityGrade = "A" | "B" | "C" | "F";

export interface QualityGate {
  grade?: QualityGrade | null;
  [key: string]: unknown;
}

export interface ModelTrainingConfig extends Record<string, unknown> {
  quality_grade?: QualityGrade | null;
  quality_gate?: QualityGate | null;
}

export interface Model {
  id: number;
  wake_word: string;
  d_prime: number | null;
  quality_grade?: QualityGrade | null;
  created_at: string;
  size_bytes: number;
}

export interface ModelConfig {
  d_prime: number | null;
  far_per_hour: number | null;
  frr: number | null;
  training_config: ModelTrainingConfig;
}

export interface ModelPerformanceResponse {
  model_name: string;
  d_prime: number | null;
  threshold: number | null;
  file_size: number;
  created_at: string;
  positive_scores: number[];
  negative_scores: number[];
  evaluation_available: boolean;
  evaluation_data: Record<string, unknown>;
}

// ── Teams ──────────────────────────────────────────────────────────────────

export type TeamMemberRole = "owner" | "admin" | "member";

export interface TeamMember {
  user_id: number;
  email: string;
  name: string;
  role: TeamMemberRole;
  invited_at: string;
  joined_at: string | null;
}

export interface Team {
  id: number;
  name: string;
  created_at: string;
  owner_id: number;
  members: TeamMember[];
}

export interface TeamListItem {
  id: number;
  name: string;
  created_at: string;
  owner_id: number;
  member_count: number;
}
