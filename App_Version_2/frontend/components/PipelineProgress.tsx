"use client";

import { useEffect, useMemo, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import type { AnalyzeEstimate } from "@/lib/types";
import { PIPELINE_STAGES } from "@/lib/types";

type Props = {
  activeStageIndex: number;
  completed: boolean;
  estimate: AnalyzeEstimate | null;
};

const BASELINE_ESTIMATE_MS = 50_000;

function buildProgressMessage(estimate: AnalyzeEstimate | null): string {
  if (!estimate) {
    return "Fetching video info…";
  }

  if (estimate.duration_seconds && estimate.duration_seconds > 0) {
    return `Video is ${estimate.duration_label} — estimated ${estimate.estimated_label}`;
  }

  return `Estimated ${estimate.estimated_label}`;
}

function ForensicScanFrame() {
  const reducedMotion = useReducedMotion();

  if (reducedMotion) {
    return (
      <div
        className="mb-6 flex h-36 items-center justify-center rounded-lg border border-border bg-background/80"
        aria-hidden
      >
        <div className="h-px w-full max-w-md bg-accent/30" />
      </div>
    );
  }

  return (
    <div
      className="relative mb-6 h-36 overflow-hidden rounded-lg border border-border bg-background/80"
      aria-hidden
    >
      <div className="absolute inset-0 bg-[linear-gradient(rgba(13,148,136,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(13,148,136,0.05)_1px,transparent_1px)] bg-[size:24px_24px]" />
      <div className="absolute inset-3 rounded-md border border-dashed border-border/80" />
      {/* Scan sweep — transform-only (translateY), never layout */}
      <motion.div
        className="pointer-events-none absolute left-0 right-0 top-0 z-10 h-[2px] transform-gpu will-change-transform"
        style={{
          background:
            "linear-gradient(90deg, transparent, rgba(13, 148, 136, 0.9), transparent)",
          boxShadow:
            "0 0 14px rgba(13, 148, 136, 0.5), 0 0 28px rgba(13, 148, 136, 0.2)",
        }}
        animate={{ y: [0, 142, 0] }}
        transition={{
          duration: 2.6,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <motion.div
        className="pointer-events-none absolute left-0 right-0 top-0 z-[5] h-10 transform-gpu will-change-transform"
        style={{
          background:
            "linear-gradient(to bottom, rgba(13, 148, 136, 0.1), transparent)",
        }}
        animate={{ y: [-40, 144, -40] }}
        transition={{
          duration: 2.6,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <p className="absolute bottom-3 left-0 right-0 text-center text-[10px] font-semibold uppercase tracking-[0.25em] text-muted">
        Scanning frames
      </p>
    </div>
  );
}

export function PipelineProgress({ activeStageIndex, completed, estimate }: Props) {
  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      <p className="mb-1 text-sm font-medium text-foreground">
        {buildProgressMessage(estimate)}
      </p>
      {estimate?.title ? (
        <p className="mb-4 line-clamp-2 text-xs text-muted">{estimate.title}</p>
      ) : (
        <div className="mb-4" />
      )}

      <ForensicScanFrame />

      <ol className="space-y-4">
        {PIPELINE_STAGES.map((stage, index) => {
          const isDone = completed || index < activeStageIndex;
          const isActive = !completed && index === activeStageIndex;
          return (
            <li key={stage.id} className="flex items-start gap-3">
              <span
                className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-mono ${
                  isDone
                    ? "bg-accent/20 text-accent"
                    : isActive
                      ? "stage-active bg-accent/10 text-accent"
                      : "bg-surface-elevated text-muted"
                }`}
              >
                {isDone ? "✓" : index + 1}
              </span>
              <div>
                <p
                  className={`text-sm font-medium ${
                    isActive ? "text-accent" : isDone ? "text-foreground" : "text-muted"
                  }`}
                >
                  {stage.label}
                </p>
                {isActive && (
                  <p className="mt-0.5 text-xs text-muted">In progress…</p>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export function usePipelineStageTimer(
  running: boolean,
  estimate: AnalyzeEstimate | null,
): number {
  const [index, setIndex] = useState(0);
  const [prevRunning, setPrevRunning] = useState(running);

  // Reset stage index when a run stops — adjusted during render instead of
  // inside an effect to avoid a cascading re-render.
  if (running !== prevRunning) {
    setPrevRunning(running);
    if (!running) {
      setIndex(0);
    }
  }

  const stageDurations = useMemo(() => {
    const estimatedMs = (estimate?.estimated_seconds_high ?? 50) * 1000;
    const factor = Math.max(1, estimatedMs / BASELINE_ESTIMATE_MS);

    const weights = [0.45, 0.2, 0.18, 0.17];
    const scaledTotal = PIPELINE_STAGES.reduce(
      (sum, stage, i) => sum + stage.durationMs * (i === 0 ? factor : Math.sqrt(factor)),
      0,
    );

    return PIPELINE_STAGES.map((stage, i) => {
      const share = weights[i] ?? 0.25;
      return Math.round(scaledTotal * share);
    });
  }, [estimate]);

  useEffect(() => {
    if (!running) {
      return;
    }

    let elapsed = 0;
    const tick = window.setInterval(() => {
      elapsed += 500;
      let accumulated = 0;
      let stageIndex = 0;
      for (let i = 0; i < stageDurations.length; i++) {
        accumulated += stageDurations[i];
        if (elapsed < accumulated) {
          stageIndex = i;
          break;
        }
        if (i === stageDurations.length - 1) {
          stageIndex = i;
        }
      }
      setIndex(stageIndex);
    }, 500);

    return () => window.clearInterval(tick);
  }, [running, stageDurations]);

  return index;
}
