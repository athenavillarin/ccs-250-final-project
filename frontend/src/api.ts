/** Typed client for the FastAPI backend. */
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type MethodKey = "classical" | "scratch" | "pretrained";

export interface MethodResult {
  method: MethodKey;
  name?: string;
  mask_png?: string;
  flooded_pct?: number;
  runtime_ms?: number;
  params_m?: number;
  error?: string;
}

export interface SegmentResponse {
  width: number;
  height: number;
  device: string;
  results: MethodResult[];
}

export interface Health {
  status: string;
  device: string;
  loaded: MethodKey[];
  errors: Record<string, string>;
}

/** Fetch JSON from the API and throw with the server's message on failure. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? res.statusText);
  return res.json();
}

/** Check backend status and loaded methods. */
export const getHealth = () => request<Health>("/health");

/** Upload an image and run the chosen methods on it. */
export function segment(file: File, methods?: MethodKey[]) {
  const body = new FormData();
  body.append("image", file);
  if (methods) body.append("methods", methods.join(","));
  return request<SegmentResponse>("/segment", { method: "POST", body });
}

export interface ResultRow {
  method: MethodKey;
  name: string;
  slot: string;
  miou: string;
  dice: string;
  pixel_acc: string;
  runtime_ms: string;
  device: string;
  params_m: string;
  train_min: string;
}

export interface Recommendation {
  method: MethodKey | null;
  explanation: string;
  table: ResultRow[];
}

/** Fetch the default best method and the main results table. */
export const getRecommendation = () => request<Recommendation>("/recommendation");
