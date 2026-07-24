import type { AnalysisResult, AnalyzeEstimate, ExamplesResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export function assetUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

export async function fetchExamples(): Promise<ExamplesResponse> {
  const response = await fetch(`${API_BASE}/examples`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load example videos");
  }
  return response.json();
}

export async function fetchAnalyzeEstimate(youtubeUrl: string): Promise<AnalyzeEstimate> {
  const response = await fetch(`${API_BASE}/analyze/estimate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ youtube_url: youtubeUrl }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail ?? "Could not estimate processing time");
  }
  return payload;
}

export async function analyzeUrl(youtubeUrl: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ youtube_url: youtubeUrl }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail ?? "Analysis failed");
  }
  return payload;
}

export function youtubeThumb(videoId: string): string {
  return `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`;
}
