import type {
  AuthResponse,
  User,
  Recording,
  UploadResponse,
  TrainingStartResponse,
  TrainingJob,
  Model,
  ModelConfig,
  ModelPerformanceResponse,
  SubscriptionResponse,
  UsageResponse,
  BillingPortalResponse,
  MessageResponse,
  Team,
  TeamListItem,
  TeamMemberRole,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";
type DownloadTokenAction = "model_download" | "training_stream";

interface DownloadTokenResponse {
  token: string;
  expires_in_seconds: number;
}

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getToken(): string | null {
  return localStorage.getItem("token");
}

function setToken(token: string): void {
  localStorage.setItem("token", token);
}

function clearToken(): void {
  localStorage.removeItem("token");
}

/**
 * Decode a JWT payload without a library. Returns null if
 * the token is malformed or not a three-segment JWT.
 */
function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const payload = atob(parts[1].replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(payload) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/**
 * Returns true if the stored JWT has an `exp` claim that has already
 * passed. Returns false if there is no exp claim (treat as valid).
 */
function isTokenExpired(): boolean {
  const token = getToken();
  if (!token) return true;

  const payload = decodeJwtPayload(token);
  if (!payload || typeof payload.exp !== "number") return false;

  // 30-second buffer so we don't race the server
  return Date.now() >= (payload.exp - 30) * 1000;
}

/**
 * Handle a 401 by clearing auth state and redirecting to login.
 */
function handleSessionExpiry(): void {
  clearToken();
  // Only redirect if we aren't already on the login page
  if (!window.location.pathname.startsWith("/login")) {
    window.location.href = "/login?expired=1";
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  // Pre-check JWT expiry before making the call
  if (getToken() && isTokenExpired()) {
    handleSessionExpiry();
    throw new ApiError(401, "Session expired");
  }

  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> | undefined),
  };

  const token = getToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      // response body wasn't JSON
    }
    if (response.status === 401) {
      handleSessionExpiry();
      throw new ApiError(401, message);
    }
    throw new ApiError(response.status, message);
  }

  const contentType = response.headers.get("content-type");
  if (contentType?.includes("application/json")) {
    return response.json();
  }
  return response as unknown as T;
}

// --- Auth ---

export async function register(
  email: string,
  password: string,
  name: string,
): Promise<AuthResponse> {
  const data = await request<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
  setToken(data.token);
  return data;
}

export async function login(
  email: string,
  password: string,
): Promise<AuthResponse> {
  const data = await request<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setToken(data.token);
  return data;
}

export async function getMe(): Promise<User> {
  return request<User>("/auth/me");
}

export function logout(): void {
  clearToken();
}

export function isAuthenticated(): boolean {
  const token = getToken();
  if (!token) return false;
  if (isTokenExpired()) {
    clearToken();
    return false;
  }
  return true;
}

async function createDownloadToken(
  action: DownloadTokenAction,
  resourceId: number,
): Promise<string> {
  const data = await request<DownloadTokenResponse>("/auth/download-token", {
    method: "POST",
    body: JSON.stringify({
      action,
      resource_id: resourceId,
    }),
  });
  return data.token;
}

export async function verifyEmail(token: string): Promise<MessageResponse> {
  return request<MessageResponse>("/auth/verify-email", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

export async function resetPassword(
  token: string,
  password: string,
): Promise<MessageResponse> {
  return request<MessageResponse>("/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ token, password }),
  });
}

export async function forgotPassword(
  email: string,
): Promise<MessageResponse> {
  return request<MessageResponse>("/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

// --- Recordings ---

export async function uploadRecording(
  file: Blob,
  wakeWord: string,
  index: number,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file, `${wakeWord}_${index}.wav`);
  formData.append("wake_word", wakeWord);

  return request<UploadResponse>("/recordings/upload", {
    method: "POST",
    body: formData,
  });
}

export async function getRecordings(
  wakeWord?: string,
): Promise<Recording[]> {
  const query = wakeWord
    ? `?wake_word=${encodeURIComponent(wakeWord)}`
    : "";
  return request<Recording[]>(`/recordings${query}`);
}

// --- Training ---

export async function startTraining(
  wakeWord: string,
  recordingIds: number[],
  epochs?: number,
): Promise<TrainingStartResponse> {
  return request<TrainingStartResponse>("/training/start", {
    method: "POST",
    body: JSON.stringify({
      wake_word: wakeWord,
      recording_ids: recordingIds,
      ...(epochs !== undefined ? { epochs } : {}),
    }),
  });
}

export async function getTrainingStatus(
  jobId: number,
): Promise<TrainingJob> {
  return request<TrainingJob>(`/training/status/${jobId}`);
}

export async function createTrainingStream(
  jobId: number,
) : Promise<EventSource> {
  const token = await createDownloadToken("training_stream", jobId);
  const url = `${BASE_URL}/training/stream/${jobId}?token=${encodeURIComponent(token)}`;
  return new EventSource(url);
}

// --- Models ---

export async function getModels(): Promise<Model[]> {
  return request<Model[]>("/models");
}

export async function getModelDownloadUrl(modelId: number): Promise<string> {
  const token = await createDownloadToken("model_download", modelId);
  return `${BASE_URL}/models/${modelId}/download?token=${encodeURIComponent(token)}`;
}

export async function getModelConfig(
  modelId: number,
): Promise<ModelConfig> {
  return request<ModelConfig>(`/models/${modelId}/config`);
}

export async function getModelPerformance(
  modelId: number,
): Promise<ModelPerformanceResponse> {
  return request<ModelPerformanceResponse>(`/models/${modelId}/performance`);
}

export async function deleteModel(modelId: number): Promise<void> {
  await request<void>(`/models/${modelId}`, {
    method: "DELETE",
  });
}

// --- Billing ---

export interface CheckoutResponse {
  checkout_url: string;
}

export async function createCheckout(
  tier: string,
): Promise<CheckoutResponse> {
  return request<CheckoutResponse>("/billing/checkout", {
    method: "POST",
    body: JSON.stringify({ tier }),
  });
}

export async function getSubscription(): Promise<SubscriptionResponse> {
  return request<SubscriptionResponse>("/billing/subscription");
}

export async function createBillingPortal(): Promise<BillingPortalResponse> {
  return request<BillingPortalResponse>("/billing/portal", {
    method: "POST",
  });
}

export async function getUsage(): Promise<UsageResponse> {
  return request<UsageResponse>("/billing/usage");
}

// --- Teams ---

export async function createTeam(
  name: string,
  description?: string,
): Promise<Team> {
  return request<Team>("/teams", {
    method: "POST",
    body: JSON.stringify({ name, ...(description ? { description } : {}) }),
  });
}

export async function listTeams(): Promise<TeamListItem[]> {
  return request<TeamListItem[]>("/teams");
}

export async function getTeam(teamId: number): Promise<Team> {
  return request<Team>(`/teams/${teamId}`);
}

export async function inviteMember(
  teamId: number,
  email: string,
  role?: TeamMemberRole,
): Promise<MessageResponse> {
  return request<MessageResponse>(`/teams/${teamId}/invite`, {
    method: "POST",
    body: JSON.stringify({ email, ...(role ? { role } : {}) }),
  });
}

export async function joinTeam(
  teamId: number,
  inviteToken: string,
): Promise<Team> {
  return request<Team>(
    `/teams/${teamId}/join?token=${encodeURIComponent(inviteToken)}`,
    { method: "POST" },
  );
}

export async function removeTeamMember(
  teamId: number,
  memberId: number,
): Promise<MessageResponse> {
  return request<MessageResponse>(`/teams/${teamId}/members/${memberId}`, {
    method: "DELETE",
  });
}

export async function shareModel(
  teamId: number,
  modelId: number,
): Promise<Model> {
  return request<Model>(`/teams/${teamId}/models/${modelId}/share`, {
    method: "POST",
  });
}

export async function listTeamModels(teamId: number): Promise<Model[]> {
  return request<Model[]>(`/teams/${teamId}/models`);
}

export async function deleteTeam(
  teamId: number,
): Promise<MessageResponse> {
  return request<MessageResponse>(`/teams/${teamId}`, {
    method: "DELETE",
  });
}

export { ApiError };
