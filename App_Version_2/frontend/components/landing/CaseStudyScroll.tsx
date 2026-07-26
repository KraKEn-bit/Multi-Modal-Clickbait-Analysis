"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ChevronLeft, ChevronRight, X, Check } from "lucide-react";
import { HARD_CASES, type HardCase } from "@/lib/hardCases";
import { assetUrl, youtubeStoryboardFrame } from "@/lib/api";
import { CaseDivergenceChart } from "@/components/charts/CaseDivergenceChart";
import { formatConfidence } from "@/lib/format";

const FRAME_META = [
  { key: "hook", label: "Hook", step: 1 },
  { key: "context", label: "Context", step: 2 },
  { key: "delivery", label: "Delivery", step: 3 },
] as const;

const SKELETON_MS = 400;

function FrameImage({
  videoId,
  url,
  frameIndex,
  label,
}: {
  videoId: string;
  url: string;
  frameIndex: 0 | 1 | 2;
  label: string;
}) {
  const primary = assetUrl(url);
  const fallback = youtubeStoryboardFrame(videoId, frameIndex);
  const [src, setSrc] = useState(primary);

  useEffect(() => {
    setSrc(primary);
  }, [primary, videoId, frameIndex]);

  return (
    /* eslint-disable-next-line @next/next/no-img-element */
    <img
      src={src}
      alt={`${label} frame`}
      className="h-full w-full object-cover"
      loading="lazy"
      onError={() => {
        setSrc((current) => (current === fallback ? current : fallback));
      }}
    />
  );
}

function CaseStudySkeleton() {
  return (
    <section id="case-study" className="scroll-mt-8 py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-6">
        <div className="skeleton h-4 w-32" />
        <div className="mt-4 skeleton h-10 w-2/3 max-w-lg" />
        <div className="mt-10 skeleton h-[640px] w-full rounded-2xl" />
      </div>
    </section>
  );
}

export function CaseStudyScroll() {
  const [ready, setReady] = useState(false);
  const [index, setIndex] = useState(0);
  const reducedMotion = useReducedMotion();

  const caseData = HARD_CASES[index] ?? HARD_CASES[0];
  const alignKeys = ["hook", "context", "delivery"] as const;
  const textWrong = caseData.text_only.verdict !== caseData.ground_truth;

  useEffect(() => {
    if (reducedMotion) {
      setReady(true);
      return;
    }
    const timer = window.setTimeout(() => setReady(true), SKELETON_MS);
    return () => window.clearTimeout(timer);
  }, [reducedMotion]);

  const go = (next: number) => {
    setIndex((next + HARD_CASES.length) % HARD_CASES.length);
  };

  if (!ready) {
    return <CaseStudySkeleton />;
  }

  return (
    <section id="case-study" className="scroll-mt-8 py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-6">
        <span className="eyebrow">Hard cases</span>
        <h2 className="mt-4 max-w-2xl text-3xl font-semibold leading-snug tracking-tight text-foreground sm:text-4xl">
          When the title alone isn&apos;t enough
        </h2>
        <p className="mt-4 max-w-2xl text-base leading-relaxed text-muted">
          Four videos where BanglaBERT misread the headline. Adding frames changed
          the verdict.
        </p>

        <div className="soft-card mt-12 overflow-hidden">
          {/* Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] px-5 py-4 sm:px-7">
            <div className="flex flex-wrap items-center gap-3">
              {HARD_CASES.map((item, i) => (
                <button
                  key={item.video_id}
                  type="button"
                  onClick={() => setIndex(i)}
                  aria-label={`Example ${i + 1}: ${item.example_label}`}
                  aria-current={index === i ? "true" : undefined}
                  className={`h-2 rounded-full transition-all duration-300 ${
                    index === i
                      ? "w-8 bg-accent"
                      : "w-2 bg-white/20 hover:bg-white/35"
                  }`}
                />
              ))}
              <span className="ml-1 text-sm text-muted">
                {index + 1} / {HARD_CASES.length}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => go(index - 1)}
                aria-label="Previous example"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.08] text-muted transition-colors hover:border-white/15 hover:text-foreground"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => go(index + 1)}
                aria-label="Next example"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.08] text-muted transition-colors hover:border-white/15 hover:text-foreground"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={caseData.video_id}
              initial={reducedMotion ? false : { opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reducedMotion ? undefined : { opacity: 0, y: -8 }}
              transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
              className="transform-gpu px-5 py-7 sm:px-7 sm:py-8"
            >
              {/* Title block */}
              <p className="text-sm font-medium text-accent">{caseData.example_label}</p>
              <h3 className="mt-2 text-lg font-semibold leading-snug text-foreground sm:text-xl">
                {caseData.title}
              </h3>
              <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">
                {caseData.example_description}
              </p>

              {/* Verdict comparison + TDS */}
              <div className="mt-7 rounded-xl border border-white/[0.06] bg-black/20 px-4 py-3 sm:px-5">
                <p className="text-xs text-muted">
                  Human label in the study:{" "}
                  <span className="font-medium text-foreground">{caseData.ground_truth}</span>
                </p>
                <p className="mt-1 text-xs leading-relaxed text-muted/90">
                  Each % is how confident that model is in its own prediction — not a
                  clickbait vs. genuine scoreboard. Correctness means matching the human label.
                </p>
              </div>

              <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1fr_140px]">
                <div className="flex gap-3 rounded-xl border border-white/[0.07] bg-black/25 p-4">
                  <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-clickbait/15">
                    <X className="h-3.5 w-3.5 text-clickbait" aria-hidden />
                  </span>
                  <div>
                    <p className="text-xs text-muted">Title-only model predicted</p>
                    <p className="mt-0.5 text-sm font-medium text-foreground">
                      {caseData.text_only.verdict}
                    </p>
                    <p className="mt-0.5 text-xs text-muted">
                      {formatConfidence(caseData.text_only.confidence)} confident in that label
                    </p>
                    {textWrong && (
                      <p className="mt-1.5 text-xs text-clickbait/80">
                        Does not match human label
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex gap-3 rounded-xl border border-agent/20 bg-agent-soft/30 p-4">
                  <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-agent/15">
                    <Check className="h-3.5 w-3.5 text-agent" aria-hidden />
                  </span>
                  <div>
                    <p className="text-xs text-muted">VTCF (title + video) predicted</p>
                    <p className="mt-0.5 text-sm font-medium text-foreground">
                      {caseData.verdict}
                    </p>
                    <p className="mt-0.5 text-xs text-muted">
                      {formatConfidence(caseData.confidence)} confident in that label
                    </p>
                    <p className="mt-1.5 text-xs text-agent/80">Matches human label</p>
                  </div>
                </div>

                <div className="flex flex-col justify-center rounded-xl border border-white/[0.07] bg-black/25 px-4 py-3 text-center lg:text-left">
                  <p className="text-xs text-muted">TDS</p>
                  <p className="mt-0.5 font-mono text-2xl font-semibold text-accent">
                    {caseData.tds_score.toFixed(3)}
                  </p>
                </div>
              </div>

              {/* Filmstrip */}
              <div className="mt-8">
                <div className="mb-4 flex items-end justify-between gap-4">
                  <p className="text-sm font-medium text-foreground">Video frames</p>
                  <p className="hidden text-xs text-muted sm:block">
                    Hook → Context → Delivery
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  {FRAME_META.map((frame, stepIndex) => {
                    const align = caseData.alignment_scores[alignKeys[stepIndex]];
                    return (
                      <div key={frame.key} className="group">
                        <div className="relative aspect-video overflow-hidden rounded-xl border border-white/[0.08] bg-black/40">
                          <FrameImage
                            videoId={caseData.video_id}
                            url={caseData.frame_urls[stepIndex]}
                            frameIndex={stepIndex as 0 | 1 | 2}
                            label={frame.label}
                          />
                          <span className="absolute left-3 top-3 flex h-6 w-6 items-center justify-center rounded-md bg-black/60 text-xs font-medium text-foreground backdrop-blur-none">
                            {frame.step}
                          </span>
                        </div>
                        <div className="mt-2.5 flex items-center justify-between gap-2">
                          <span className="text-sm text-foreground">{frame.label}</span>
                          <span className="font-mono text-[11px] text-muted">
                            {align >= 0 ? "+" : ""}
                            {align.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Chart */}
              <div className="mt-8 border-t border-white/[0.06] pt-8">
                <p className="text-sm font-medium text-foreground">Alignment over time</p>
                <p className="mt-1 text-xs text-muted">
                  How closely each frame matches the headline promise.
                </p>
                <div className="mt-5 rounded-xl border border-white/[0.06] bg-black/20 p-4 sm:p-5">
                  <CaseDivergenceChart alignment={caseData.alignment_scores} />
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}
