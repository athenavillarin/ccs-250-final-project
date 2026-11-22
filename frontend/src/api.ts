/** Typed client for the FastAPI backend. */
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type MethodKey = "classical" | "scratch" | "pretrained";

/** Display name and a short kind label per method. */
export const METHOD_INFO: Record<MethodKey, { name: string; kind: string }> = {
  pretrained: { name: "SegFormer-B0", kind: "Pretrained transformer, fine-tuned" },
  scratch: { name: "U-Net", kind: "Neural network trained from scratch" },
  classical: { name: "Graph cut", kind: "Classical, no neural network" },
};

export interface MethodResult {
  method: MethodKey;
  mask_png?: string;
  flooded_pct?: number;
  runtime_ms?: number;
  error?: string;
}

interface SegmentResponse {
  results: MethodResult[];
}

/** Fetch JSON from the API and throw with the server's message on failure. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? res.statusText);
  return res.json();
}

/** Resolve once the backend answers, which also wakes a sleeping server. */
export const getHealth = () => request<{ status: string }>("/health");

/** Upload an image and run one method on it; the signal cancels the request. */
export async function segment(file: File, method: MethodKey, signal?: AbortSignal): Promise<MethodResult> {
  const body = new FormData();
  body.append("image", file);
  body.append("methods", method);
  return (await request<SegmentResponse>("/segment", { method: "POST", body, signal })).results[0];
}

/** Key of the method with the best test score, or null if results are unavailable. */
export const getRecommended = async () => (await request<{ method: MethodKey | null }>("/recommendation")).method;
