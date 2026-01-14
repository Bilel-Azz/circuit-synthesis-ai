/**
 * API Client for Circuit Synthesis Backend
 *
 * MODIFY API_BASE_URL if your backend is hosted elsewhere
 */

// Use proxy route to avoid CORS issues (Vercel HTTPS -> OVH HTTP)
const API_BASE_URL = typeof window !== 'undefined'
  ? '/api/proxy'  // Client-side: use proxy
  : (process.env.BACKEND_URL || 'http://57.128.57.31:8000');  // Server-side: direct

export interface Component {
  type: string;
  type_id: number;
  node_a: number;
  node_b: number;
  value: number;
  formatted_value: string;
}

export interface Impedance {
  magnitude: number[];
  phase: number[];
  frequencies: number[];
}

export interface ErrorMetrics {
  magnitude: number;
  phase: number;
  total: number;
}

export interface Candidate {
  components: Component[];
  impedance: Impedance;
  error: ErrorMetrics;
}

export interface GenerationStats {
  total: number;
  valid: number;
  invalid: number;
  empty: number;
  compute_error: number;
}

export interface GenerateResponse {
  success: boolean;
  message?: string;
  best?: Candidate;
  candidates: Candidate[];
  num_candidates: number;
  stats?: GenerationStats;
}

export interface Config {
  num_freq: number;
  freq_min: number;
  freq_max: number;
  frequencies: number[];
}

export interface HealthCheck {
  status: string;
  model_loaded: boolean;
  device: string;
  current_model?: string;
  current_model_name?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  available: boolean;
  is_current: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  current: string;
}

export async function checkHealth(): Promise<HealthCheck> {
  const res = await fetch(`${API_BASE_URL}/`);
  if (!res.ok) throw new Error('Backend not available');
  return res.json();
}

export async function getConfig(): Promise<Config> {
  const res = await fetch(`${API_BASE_URL}/config`);
  if (!res.ok) throw new Error('Failed to get config');
  return res.json();
}

export async function generateCircuit(
  magnitude: number[],
  phase: number[],
  options?: { tau?: number; num_candidates?: number; model_id?: string }
): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      magnitude,
      phase,
      tau: options?.tau ?? 0.5,
      num_candidates: options?.num_candidates ?? 10,
      model_id: options?.model_id,
    }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Generation failed');
  }
  return res.json();
}

export async function getModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_BASE_URL}/models`);
  if (!res.ok) throw new Error('Failed to get models');
  return res.json();
}

export async function switchModel(model_id: string): Promise<{ success: boolean; model_id: string; name: string }> {
  const res = await fetch(`${API_BASE_URL}/models/switch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Failed to switch model');
  }
  return res.json();
}

export async function computeImpedance(components: Component[]): Promise<Impedance> {
  const res = await fetch(`${API_BASE_URL}/compute-impedance`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(components),
  });
  if (!res.ok) throw new Error('Failed to compute impedance');
  return res.json();
}
