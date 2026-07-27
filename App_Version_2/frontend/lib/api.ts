import type { AnalysisResult, AnalyzeEstimate, ExamplesResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export function getApiBase(): string {
  return API_BASE;
}

/**
 * True when the browser page cannot call the configured API.
 * Typical case: HTTPS Vercel site + default http://127.0.0.1:8000
 * (mixed content / unreachable host → "Failed to fetch").
 */
export function isLiveApiUnreachableFromPage(): boolean {
  if (typeof window === "undefined") return false;
  const pageHost = window.location.hostname;
  const localPage = pageHost === "localhost" || pageHost === "127.0.0.1";
  if (localPage) return false;

  let apiHost = "";
  let apiProtocol = "";
  try {
    const api = new URL(API_BASE);
    apiHost = api.hostname;
    apiProtocol = api.protocol;
  } catch {
    return true;
  }

  const apiIsLoopback = apiHost === "localhost" || apiHost === "127.0.0.1";
  if (apiIsLoopback) return true;
  if (window.location.protocol === "https:" && apiProtocol === "http:") return true;
  return false;
}

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
    const detail = payload.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ")
          : "Analysis failed";
    throw new Error(message || "Analysis failed");
  }
  return payload;
}

export function youtubeThumb(videoId: string): string {
  return `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`;
}

/** Distinct storyboard keyframes (1–3) when cached PNGs are unavailable. */
export function youtubeStoryboardFrame(
  videoId: string,
  index: 0 | 1 | 2,
): string {
  return `https://img.youtube.com/vi/${videoId}/${index + 1}.jpg`;
}
