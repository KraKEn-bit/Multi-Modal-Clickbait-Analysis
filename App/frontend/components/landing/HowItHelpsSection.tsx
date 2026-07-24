"use client";

import type { AnalysisResult } from "@/lib/types";
import { assetUrl } from "@/lib/api";
import { FRAME_LABELS } from "@/lib/types";
import { explanationMatchesVerdict, getDisplayExplanation } from "@/lib/previewExample";
import { formatConfidence } from "@/lib/format";
import { ScrollReveal, ScrollRevealItem } from "@/components/ScrollReveal";
import { Film, LineChart, MessageSquareText } from "lucide-react";

const BULLETS = [
  {
    icon: Film,
    text: "See the exact hook, context, and delivery frames the model analyzed",
  },
  {
    icon: LineChart,
    text: "Temporal Divergence Score (TDS) — quantifies how much the video's visuals shift from start to end",
  },
  {
    icon: MessageSquareText,
    text: "A plain-language explanation for every verdict",
  },
] as const;

type Props = {
  previewExample: AnalysisResult | null;
};

export function ResultsPreview({ example }: { example: AnalysisResult }) {
  const isClickbait = example.verdict === "CLICKBAIT";
  const verdictColor = isClickbait ? "text-clickbait" : "text-genuine";
  const verdictBg = isClickbait
    ? "bg-clickbait/10 border-clickbait/30"
    : "bg-genuine/10 border-genuine/30";

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-surface shadow-[0_24px_80px_rgba(0,0,0,0.35)]">
      <div className="flex items-center justify-between gap-2 border-b border-border bg-surface-elevated/60 px-5 py-2">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-accent">
          Cached example result
        </p>
        {!explanationMatchesVerdict(example) && (
          <p className="text-[10px] text-muted">Illustrative UI preview</p>
        )}
      </div>
      <div className={`border-b border-border px-5 py-4 ${verdictBg}`}>
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted">
          Full VTCF verdict
        </p>
        <p className={`mt-1 font-mono text-2xl font-bold ${verdictColor}`}>{example.verdict}</p>
        <p className="mt-1 text-xs text-muted">
          {formatConfidence(example.confidence)} confidence
        </p>
      </div>

      <div className="grid grid-cols-3 gap-px bg-border">
        {example.frame_urls.map((url, index) => (
          <figure key={url} className="bg-surface">
            <img
              src={assetUrl(url)}
              alt={FRAME_LABELS[index]}
              className="aspect-video w-full object-cover"
            />
            <figcaption className="px-2 py-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-accent">
                {FRAME_LABELS[index]}
              </p>
            </figcaption>
          </figure>
        ))}
      </div>

      <div className="border-t border-border px-5 py-4">
        <div className="flex items-end justify-between gap-3">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted">
              Temporal Divergence Score (TDS)
            </p>
            <p className="font-mono text-xl font-bold">
              {example.tds_score.toFixed(3)}
              <span className="ml-1 text-sm font-normal text-muted">/ 1.0</span>
            </p>
          </div>
          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-surface-elevated">
            <div
              className="h-full rounded-full bg-accent"
              style={{ width: `${Math.min(example.tds_score * 100, 100)}%` }}
            />
          </div>
        </div>
        <p className="mt-3 line-clamp-2 text-xs leading-relaxed text-muted">
          {getDisplayExplanation(example)}
        </p>
      </div>
    </div>
  );
}

export function HowItHelpsSection({ previewExample }: Props) {
  return (
    <ScrollReveal
      id="how-it-helps"
      className="scroll-mt-8 border-t border-border/50 py-16 sm:py-20"
    >
      <div className="mx-auto max-w-6xl">
        <ScrollRevealItem>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted">
            Transparency
          </p>
          <h2 className="mt-3 max-w-2xl font-[family-name:var(--font-fraunces)] text-2xl font-bold tracking-tight sm:text-4xl">
            See the evidence, not just a verdict.
          </h2>
        </ScrollRevealItem>

        <div className="mt-12 grid items-center gap-10 lg:grid-cols-2 lg:gap-14">
          <ScrollRevealItem delay={0.08}>
            {previewExample ? (
              <ResultsPreview example={previewExample} />
            ) : (
              <div className="flex h-64 items-center justify-center rounded-2xl border border-dashed border-border bg-surface/50 text-sm text-muted">
                Loading preview…
              </div>
            )}
          </ScrollRevealItem>

          <ul className="space-y-6">
            {BULLETS.map(({ icon: Icon, text }, index) => (
              <ScrollRevealItem key={text} delay={0.12 + index * 0.08}>
                <li className="flex gap-4">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-accent/25 bg-accent/10">
                    <Icon className="h-5 w-5 text-accent" aria-hidden />
                  </span>
                  <p className="pt-1.5 text-base leading-relaxed text-foreground">{text}</p>
                </li>
              </ScrollRevealItem>
            ))}
          </ul>
        </div>
      </div>
    </ScrollReveal>
  );
}
