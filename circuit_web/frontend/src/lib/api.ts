/**
 * API Client for Circuit Synthesis Backend
 *
 * MODIFY API_BASE_URL if your backend is hosted elsewhere
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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

export interface GenerateResponse {
  success: boolean;
  message?: string;
  best?: Candidate;
  candidates: Candidate[];
  num_candidates: number;
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
  options?: { tau?: number; num_candidates?: number }
): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      magnitude,
      phase,
      tau: options?.tau ?? 0.5,
      num_candidates: options?.num_candidates ?? 10,
    }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Generation failed');
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
