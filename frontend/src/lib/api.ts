export type ComponentKind = "stt" | "llm" | "tts";
export type SessionStatus = "warming" | "ready" | "error" | "closed";

export interface ModelManifest {
  id: string;
  label: string;
  component: ComponentKind;
  provider: string;
  path?: string | null;
  source?: string | null;
  description?: string | null;
  config: Record<string, unknown>;
}

export interface HardwareProfile {
  cpu_model: string;
  cpu_threads: number;
  gpu_name: string;
  gpu_target: string;
  llm_quantization: string;
  notes: string[];
}

export interface ModelCatalog {
  stt: ModelManifest[];
  llm: ModelManifest[];
  tts: ModelManifest[];
  hardware: HardwareProfile;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  created_at: string;
}

export interface SessionState {
  session_id: string;
  stt_model: ModelManifest;
  llm_model: ModelManifest;
  tts_model: ModelManifest;
  system_prompt: string;
  status: SessionStatus;
  created_at: string;
  updated_at: string;
  conversation_history: ChatMessage[];
  last_error?: string | null;
}

export interface SessionListItem {
  session_id: string;
  title: string;
  preview: string;
  updated_at: string;
  status: SessionStatus;
  is_live: boolean;
}

export interface TurnMetrics {
  stt_ms: number;
  llm_ms: number;
  tts_ms: number;
  pipeline_ms: number;
  heard_to_response_ms?: number | null;
}

export interface TurnResponse {
  session_id: string;
  transcript: string;
  assistant_text: string;
  assistant_audio_url: string;
  conversation_history: ChatMessage[];
  metrics: TurnMetrics;
}

export interface PipelineStartResponse {
  session: SessionState;
}

export interface SessionSummary {
  session: SessionState;
}

export interface LiveMetrics {
  gpu_memory_used_mb?: number | null;
  gpu_memory_total_mb?: number | null;
  gpu_memory_free_mb?: number | null;
  gpu_memory_percent?: number | null;
  gpu_name?: string | null;
  updated_at: string;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export function getApiBase(): string {
  return API_BASE;
}

export function fetchModels(): Promise<ModelCatalog> {
  return request<ModelCatalog>("/api/models");
}

export function fetchSessions(): Promise<SessionListItem[]> {
  return request<SessionListItem[]>("/api/sessions");
}

export function fetchSession(sessionId: string): Promise<SessionSummary> {
  return request<SessionSummary>(`/api/sessions/${sessionId}`);
}

export function fetchLiveMetrics(): Promise<LiveMetrics> {
  return request<LiveMetrics>("/api/metrics/live");
}

export function startPipeline(payload: {
  stt_model_id: string;
  llm_model_id: string;
  tts_model_id: string;
  system_prompt: string;
}): Promise<PipelineStartResponse> {
  return request<PipelineStartResponse>("/api/pipeline/start", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function stopPipeline(sessionId: string): Promise<SessionSummary> {
  return request<SessionSummary>(`/api/sessions/${sessionId}/stop`, {
    method: "POST",
  });
}

export function sendTextTurn(
  sessionId: string,
  text: string,
  signal?: AbortSignal,
): Promise<TurnResponse> {
  return request<TurnResponse>(`/api/sessions/${sessionId}/turns/text`, {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });
}

export async function sendAudioTurn(
  sessionId: string,
  audio: Blob,
  signal?: AbortSignal,
): Promise<TurnResponse> {
  const formData = new FormData();
  formData.append("audio", audio, "microphone.wav");

  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/turns/audio`, {
    method: "POST",
    signal,
    body: formData,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail ?? "Upload failed");
  }
  return response.json() as Promise<TurnResponse>;
}
