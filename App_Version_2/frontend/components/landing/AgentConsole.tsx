"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { createTimeline } from "animejs";
import { ArrowDown, CheckCircle2, Loader2, Sparkles } from "lucide-react";
import type { AnalysisResult, AnalyzeEstimate } from "@/lib/types";
import { getDisplayExplanation } from "@/lib/previewExample";

/**
 * Functional VTCF agent console (manus.im style).
 *
 * - Paste a URL and the real pipeline runs; Anime.js cascades the
 *   staggered stage lines in while `stageIndex` (wall-clock synced
 *   upstream) drives pending → active → done states.
 * - On live-fetch failure the page flips `fallback` on: a kokonut
 *   skeleton shimmers while the cached hard case loads and a toast
 *   explains the switch.
 * - When a result lands the agent "types" its analysis (mono), then
 *   the verdict panel expands via Framer Motion.
 */

const STAGE_LABELS = [
  "[FETCHING METADATA]",
  "[SAMPLING FRAMES]",
  "[ENCODING BANGLABERT]",
  "[CALCULATING TDS]",
] as const;

const TYPE_INTERVAL_MS = 16;
const CHARS_PER_TICK = 2;
const MAX_TYPED_CHARS = 220;

type Props = {
  url: string;
  onUrlChange: (url: string) => void;
  onAnalyze: () => void;
  onLoadDemo: () => void;
  onViewReport: () => void;
  loading: boolean;
  fallback: boolean;
  error: string | null;
  stageIndex: number;
  estimate: AnalyzeEstimate | null;
  result: AnalysisResult | null;
  hasDemo: boolean;
};

function buildAgentSummary(result: AnalysisResult): string {
  const text = getDisplayExplanation(result);
  if (text.length <= MAX_TYPED_CHARS) return text;
  return `${text.slice(0, MAX_TYPED_CHARS).trimEnd()}…`;
}

export function AgentConsole({
  url,
  onUrlChange,
  onAnalyze,
  onLoadDemo,
  onViewReport,
  loading,
  fallback,
  error,
  stageIndex,
  estimate,
  result,
  hasDemo,
}: Props) {
  const reducedMotion = useReducedMotion();
  const stagesRef = useRef<HTMLDivElement>(null);
  const [typedCount, setTypedCount] = useState(0);

  const status = fallback
    ? "FALLBACK"
    : loading
      ? "RUNNING"
      : result
        ? "DONE"
        : "IDLE";

  const summary = result ? buildAgentSummary(result) : "";
  const typingDone = reducedMotion || typedCount >= summary.length;

  // Anime.js — staggered cascade of the stage lines when a run starts.
  useEffect(() => {
    if (!loading || reducedMotion) return;
    const root = stagesRef.current;
    if (!root) return;

    const lines = Array.from(root.querySelectorAll<HTMLElement>("[data-stage-line]"));
    const timeline = createTimeline({
      defaults: { ease: "outCubic", duration: 340 },
    });
    lines.forEach((line, i) => {
      timeline.add(line, { opacity: [0, 1], translateX: [-10, 0] }, i * 130);
    });

    return () => {
      timeline.pause();
    };
  }, [loading, reducedMotion]);

  // Agent typing effect — runs each time a new result lands.
  const resultId = result?.video_id ?? null;
  useEffect(() => {
    setTypedCount(0);
    if (!resultId || reducedMotion) return;
    const timer = window.setInterval(() => {
      setTypedCount((count) => {
        const next = count + CHARS_PER_TICK;
        if (next >= MAX_TYPED_CHARS + 10) {
          window.clearInterval(timer);
        }
        return next;
      });
    }, TYPE_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [resultId, reducedMotion]);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (!loading && !fallback) onAnalyze();
  };

  const isClickbait = result?.verdict === "CLICKBAIT";

  return (
    <div className="mx-auto w-full max-w-2xl text-left">
      <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#0d0d10] shadow-[0_2px_10px_rgba(0,0,0,0.5),0_40px_90px_-30px_rgba(0,229,255,0.12)]">
        {/* Terminal header strip */}
        <div className="flex items-center justify-between border-b border-white/[0.06] px-5 py-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5" aria-hidden>
              <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
              <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
              <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
            </div>
            <p className="font-mono text-xs font-medium text-zinc-300">
              vtcf-agent
            </p>
          </div>
          <span
            className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-[10px] font-medium tracking-wider ${
              status === "DONE"
                ? "bg-agent-soft text-agent"
                : status === "IDLE"
                  ? "bg-white/5 text-zinc-400"
                  : "bg-accent-soft text-accent"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                status === "RUNNING" || status === "FALLBACK"
                  ? "animate-pulse bg-accent"
                  : status === "DONE"
                    ? "bg-agent"
                    : "bg-zinc-500"
              }`}
              aria-hidden
            />
            {status}
          </span>
        </div>

        <div className="p-5 sm:p-6">
          {/* Live URL input */}
          <form onSubmit={handleSubmit} className="flex items-center gap-3">
            <input
              type="url"
              value={url}
              onChange={(event) => onUrlChange(event.target.value)}
              placeholder="https://youtube.com/watch?v=…"
              disabled={loading || fallback}
              className="console-input min-w-0 flex-1 rounded-xl px-4 py-3 font-mono text-sm text-foreground disabled:opacity-60"
              aria-label="YouTube URL to analyze"
            />
            <button
              type="submit"
              disabled={loading || fallback || !url.trim()}
              className="flex shrink-0 items-center gap-2 rounded-xl bg-agent px-5 py-3 text-sm font-semibold text-[#04120c] shadow-[0_8px_24px_-8px_rgba(0,255,163,0.5)] transition-colors duration-200 hover:bg-[#4dffbe] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading || fallback ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              ) : (
                <Sparkles className="h-4 w-4" aria-hidden />
              )}
              Analyze
            </button>
          </form>

          {/* Idle prompt */}
          {status === "IDLE" && !error && (
            <div className="mt-5 font-mono text-xs leading-relaxed text-muted">
              <p>
                <span className="text-agent">$</span> paste a Bangla YouTube URL —
                I&apos;ll watch it before you do.
              </p>
              {hasDemo && (
                <button
                  type="button"
                  onClick={onLoadDemo}
                  className="mt-2 text-accent transition-colors duration-200 hover:text-accent-strong"
                >
                  → or run a cached hard case from the 805-video test set
                </button>
              )}
            </div>
          )}

          {/* Error line */}
          {status === "IDLE" && error && (
            <div className="mt-5 space-y-2 font-mono text-xs leading-relaxed">
              <p className="text-clickbait">! {error}</p>
              {hasDemo && (
                <button
                  type="button"
                  onClick={onLoadDemo}
                  className="text-accent transition-colors duration-200 hover:text-accent-strong"
                >
                  → or open a cached hard case instead
                </button>
              )}
            </div>
          )}

          {/* Stage cascade — Anime.js entrance, stageIndex drives states */}
          {loading && (
            <div ref={stagesRef} className="mt-5 space-y-2">
              {STAGE_LABELS.map((label, i) => {
                const done = i < stageIndex;
                const active = i === stageIndex;
                return (
                  <div
                    key={label}
                    data-stage-line
                    className="flex transform-gpu items-center gap-3 font-mono text-xs will-change-transform"
                    style={reducedMotion ? undefined : { opacity: 0 }}
                  >
                    {done ? (
                      <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-agent" aria-hidden />
                    ) : active ? (
                      <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" aria-hidden />
                    ) : (
                      <span className="h-3.5 w-3.5 shrink-0 rounded-full border border-white/15" aria-hidden />
                    )}
                    <span
                      className={
                        done
                          ? "text-agent"
                          : active
                            ? "text-accent"
                            : "text-zinc-600"
                      }
                    >
                      {label}
                    </span>
                    {active && <span className="typing-caret" aria-hidden />}
                  </div>
                );
              })}
              {estimate && (
                <p className="pt-1 font-mono text-[11px] text-zinc-600">
                  {"// "}estimated {estimate.estimated_label}
                  {estimate.duration_label ? ` · video ${estimate.duration_label}` : ""}
                </p>
              )}
            </div>
          )}

          {/* Fallback — kokonut skeleton while the cached case loads */}
          {fallback && (
            <div className="mt-5 space-y-2.5" aria-live="polite">
              <p className="font-mono text-[11px] text-warning">
                {"// "}live fetch timed out — retrieving cached hard case
              </p>
              <div className="skeleton h-4 w-3/4" />
              <div className="skeleton h-4 w-1/2" />
              <div className="skeleton h-16 w-full" />
            </div>
          )}

          {/* Agent's typed read-out */}
          {result && !loading && !fallback && (
            <div className="mt-5 rounded-xl bg-white/[0.03] p-4">
              <p className="font-mono text-xs leading-relaxed text-zinc-300">
                <span className="text-agent">$</span>{" "}
                {reducedMotion ? summary : summary.slice(0, typedCount)}
                {!typingDone && <span className="typing-caret" aria-hidden />}
              </p>
            </div>
          )}

          {/* Verdict panel — expands once the agent finishes typing */}
          <AnimatePresence initial={false}>
            {result && !loading && !fallback && typingDone && (
              <motion.div
                key={`verdict-${result.video_id}`}
                initial={reducedMotion ? false : { height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={reducedMotion ? undefined : { height: 0, opacity: 0 }}
                transition={{ duration: 0.45, ease: [0.32, 0.72, 0, 1] }}
                className="overflow-hidden"
              >
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/[0.07] p-4">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span
                      className={`rounded-full px-3 py-1 font-mono text-xs font-bold ${
                        isClickbait
                          ? "bg-clickbait/15 text-clickbait"
                          : "bg-agent-soft text-agent"
                      }`}
                    >
                      {result.verdict} · {result.confidence.toFixed(1)}%
                    </span>
                    <span className="data-point">
                      TDS {result.tds_score.toFixed(2)} / 1.0
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={onViewReport}
                    className="flex items-center gap-1.5 font-mono text-xs font-medium text-accent transition-colors duration-200 hover:text-accent-strong"
                  >
                    view full report
                    <ArrowDown className="h-3.5 w-3.5" aria-hidden />
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
