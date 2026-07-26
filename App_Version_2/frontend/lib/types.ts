export type AnalyzeEstimate = {
  video_id: string;
  youtube_url: string;
  title: string;
  duration_seconds: number | null;
  duration_label: string;
  estimated_seconds_low: number;
  estimated_seconds_high: number;
  estimated_label: string;
};

export type AlignmentScores = {
  hook: number;
  context: number;
  delivery: number;
};

export type TextOnlyResult = {
  verdict: string;
  confidence: number;
  wrong?: boolean;
};

export type AnalysisResult = {
  video_id: string;
  youtube_url: string;
  title: string;
  verdict: "CLICKBAIT" | "GENUINE";
  confidence: number;
  tds_score: number;
  explanation: string;
  alignment_scores: AlignmentScores;
  frame_urls: string[];
  processing_time_seconds: number;
  category?: string;
  example_label?: string;
  example_description?: string;
  ground_truth?: string;
  text_only?: TextOnlyResult;
  vtcf_rescued?: boolean;
};

export type ExamplesResponse = {
  examples: AnalysisResult[];
  research_note: string;
};

export type PipelineStage = {
  id: string;
  label: string;
  durationMs: number;
};

export const PIPELINE_STAGES: PipelineStage[] = [
  { id: "downloading", label: "Downloading video stream from YouTube", durationMs: 18000 },
  { id: "detecting", label: "Detecting scene boundaries (PySceneDetect)", durationMs: 12000 },
  { id: "extracting", label: "Extracting hook, context, and delivery frames", durationMs: 10000 },
  { id: "inferring", label: "Running BanglaBERT + ViT fusion model", durationMs: 8000 },
];

export const FRAME_LABELS = ["Hook", "Context", "Delivery"] as const;
