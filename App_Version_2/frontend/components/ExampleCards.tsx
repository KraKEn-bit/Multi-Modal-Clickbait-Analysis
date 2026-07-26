"use client";

import { SearchCheck } from "lucide-react";
import type { AnalysisResult } from "@/lib/types";
import { youtubeThumb } from "@/lib/api";

type Props = {
  examples: AnalysisResult[];
  onSelect: (example: AnalysisResult) => void;
  disabled?: boolean;
};

const CATEGORY_STYLES: Record<string, string> = {
  clickbait: "bg-clickbait/10 text-clickbait",
  genuine: "bg-accent-soft text-accent-strong",
  hard: "bg-warning/10 text-warning",
};

function ExampleCard({
  example,
  disabled,
  onSelect,
}: {
  example: AnalysisResult;
  disabled?: boolean;
  onSelect: (example: AnalysisResult) => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onSelect(example)}
      className="soft-card soft-card--hover group relative overflow-hidden p-4 text-left disabled:pointer-events-none disabled:opacity-50"
    >
      <div className="flex gap-4">
        <div className="relative h-20 w-32 shrink-0 overflow-hidden rounded-xl">
          <img
            src={youtubeThumb(example.video_id)}
            alt=""
            className="h-full w-full transform-gpu object-cover transition-transform duration-300 ease-out group-hover:scale-105"
          />

          {/* Pure-CSS hover overlay — opacity/transform only */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent opacity-0 transition-opacity duration-300 ease-out group-hover:opacity-100" />
          <div className="absolute inset-x-0 bottom-0 flex translate-y-2 transform-gpu items-center justify-center gap-1.5 pb-2 text-[11px] font-semibold text-white opacity-0 transition-all duration-300 ease-out group-hover:translate-y-0 group-hover:opacity-100">
            <SearchCheck className="h-3.5 w-3.5 text-agent" aria-hidden />
            View analysis
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold ${
                CATEGORY_STYLES[example.category ?? "hard"]
              }`}
            >
              {example.example_label ?? example.category}
            </span>
            {example.vtcf_rescued && (
              <span className="rounded-full bg-accent-soft px-2.5 py-0.5 text-[10px] font-bold text-accent-strong">
                VTCF rescue
              </span>
            )}
          </div>
          <p className="line-clamp-2 text-sm font-semibold leading-snug text-foreground">
            {example.title}
          </p>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            {example.example_description ?? "Pre-computed — loads instantly"}
          </p>
        </div>
      </div>
    </button>
  );
}

export function ExampleCards({ examples, onSelect, disabled }: Props) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {examples.map((example) => (
        <ExampleCard
          key={example.video_id}
          example={example}
          onSelect={onSelect}
          disabled={disabled}
        />
      ))}
    </div>
  );
}
