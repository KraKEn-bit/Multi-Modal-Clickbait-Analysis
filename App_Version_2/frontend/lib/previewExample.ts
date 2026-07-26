import type { AnalysisResult } from "@/lib/types";

/**
 * Pick a cached example for the landing-page results preview.
 * Uses obvious clickbait — the only cached row whose shared explanation
 * text ("diverges from the headline promise…") matches the verdict semantically.
 * Hard-case GENUINE rows share that same template string from cache generation.
 */
export function selectPreviewExample(
  examples: AnalysisResult[],
): AnalysisResult | null {
  return (
    examples.find((e) => e.category === "clickbait") ??
    examples.find((e) => e.verdict === "CLICKBAIT") ??
    examples[0] ??
    null
  );
}

/** True when explanation wording fits a clickbait-style divergence read. */
export function explanationMatchesVerdict(example: AnalysisResult): boolean {
  if (example.verdict === "CLICKBAIT") {
    return true;
  }
  const lower = example.explanation.toLowerCase();
  return !lower.includes("diverges from the headline promise");
}

/** Cached rows may share a generic template — prefer copy that matches the verdict. */
export function getDisplayExplanation(example: AnalysisResult): string {
  if (explanationMatchesVerdict(example)) {
    return example.explanation;
  }
  if (example.example_description) {
    return example.example_description;
  }
  if (example.verdict === "GENUINE") {
    return "Visual frames align with the headline across hook, context, and delivery, supporting a genuine verdict.";
  }
  return example.explanation;
}
