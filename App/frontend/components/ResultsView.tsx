"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { AnalysisResult } from "@/lib/types";
import { assetUrl } from "@/lib/api";
import { FRAME_LABELS } from "@/lib/types";
import { getDisplayExplanation } from "@/lib/previewExample";
import { formatConfidence } from "@/lib/format";
import {
  motionTransition,
  SPRING_BOUNCY,
  SPRING_DROP,
  SPRING_EXPAND,
  SPRING_GENTLE,
  SPRING_SNAPPY,
} from "@/lib/motion";

type Props = {
  result: AnalysisResult;
  researchNote?: string;
};

const detectiveContainer = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.2,
      delayChildren: 0.04,
    },
  },
};

const verdictDrop = {
  hidden: { opacity: 0, y: -40, scale: 0.9 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: SPRING_DROP,
  },
};

const hardCaseReveal = {
  hidden: { opacity: 0, y: 20, scale: 0.97 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: SPRING_GENTLE,
  },
};

const framesStage = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.06,
    },
  },
};

const frameFan = {
  hidden: { opacity: 0, x: -52, scale: 0.86, rotate: -2.5 },
  visible: {
    opacity: 1,
    x: 0,
    scale: 1,
    rotate: 0,
    transition: SPRING_GENTLE,
  },
};

const tdsExpand = {
  hidden: { opacity: 0, y: 32, scaleY: 0.88 },
  visible: {
    opacity: 1,
    y: 0,
    scaleY: 1,
    transition: SPRING_EXPAND,
  },
};

const panelReveal = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: SPRING_GENTLE,
  },
};

function WordReveal({ text, startDelay = 0 }: { text: string; startDelay?: number }) {
  const reducedMotion = useReducedMotion();
  const words = text.split(/\s+/);

  if (reducedMotion) {
    return <>{text}</>;
  }

  return (
    <>
      {words.map((word, index) => (
        <motion.span
          key={`${word}-${index}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            ...SPRING_SNAPPY,
            delay: startDelay + index * 0.045,
          }}
          className="inline-block"
        >
          {word}
          {index < words.length - 1 ? "\u00A0" : ""}
        </motion.span>
      ))}
    </>
  );
}

export function ResultsView({ result, researchNote }: Props) {
  const reducedMotion = useReducedMotion();
  const isClickbait = result.verdict === "CLICKBAIT";
  const verdictColor = isClickbait ? "text-clickbait" : "text-genuine";
  const verdictBg = isClickbait ? "bg-clickbait/10 border-clickbait/30" : "bg-genuine/10 border-genuine/30";
  const strokeColor = isClickbait ? "#F0654A" : "#00D2FF";
  const motionInitial = reducedMotion ? false : ("hidden" as const);
  const motionAnimate = reducedMotion ? undefined : ("visible" as const);

  return (
    <motion.div
      className="space-y-8"
      variants={reducedMotion ? undefined : detectiveContainer}
      initial={motionInitial}
      animate={motionAnimate}
    >
      <motion.div
        layout
        variants={reducedMotion ? undefined : verdictDrop}
        className={`rounded-2xl border p-8 ${verdictBg}`}
      >
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted">
          Full VTCF verdict
        </p>
        <motion.p
          className={`mt-2 font-mono text-5xl font-bold tracking-tight ${verdictColor}`}
          initial={reducedMotion ? false : { scale: 1.12, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={motionTransition(!!reducedMotion, SPRING_BOUNCY)}
        >
          {result.verdict}
        </motion.p>
        <p className="mt-2 text-lg text-foreground">
          {formatConfidence(result.confidence)} confidence
          {result.processing_time_seconds > 0 && (
            <span className="ml-3 text-sm text-muted">
              · {result.processing_time_seconds}s
            </span>
          )}
        </p>
        <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted">{result.title}</p>
      </motion.div>

      {result.category === "hard" && result.text_only && (
        <motion.section
          layout
          variants={reducedMotion ? undefined : hardCaseReveal}
          className="rounded-2xl border-2 border-accent/40 bg-accent/5 p-6"
        >
          <HardCaseContent result={result} />
        </motion.section>
      )}

      <motion.div layout>
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted">
          Temporal frames
        </h3>
        <div className="evidence-board relative">
          <svg
            className="pointer-events-none absolute inset-x-0 top-[38%] hidden h-8 w-full md:block"
            viewBox="0 0 100 4"
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            <motion.path
              d="M 8 2 L 50 2 L 92 2"
              fill="none"
              stroke={strokeColor}
              strokeWidth="0.15"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={reducedMotion ? { pathLength: 1 } : { pathLength: 0, opacity: 0.4 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={motionTransition(!!reducedMotion, {
                type: "spring",
                stiffness: 70,
                damping: 18,
              })}
            />
            {[8, 50, 92].map((cx, index) => (
              <motion.circle
                key={cx}
                cx={cx}
                cy={2}
                r={0.35}
                fill={strokeColor}
                initial={reducedMotion ? { opacity: 0.85, scale: 1 } : { opacity: 0, scale: 0 }}
                animate={{ opacity: 0.85, scale: 1 }}
                transition={motionTransition(!!reducedMotion, {
                  ...SPRING_BOUNCY,
                  delay: 0.15 + index * 0.1,
                })}
              />
            ))}
          </svg>

          <motion.div
            className="grid gap-4 md:grid-cols-3"
            variants={reducedMotion ? undefined : framesStage}
          >
            {result.frame_urls.map((url, index) => (
              <motion.figure
                key={url}
                layout
                variants={reducedMotion ? undefined : frameFan}
                className="relative z-[1] overflow-hidden rounded-xl border border-border bg-surface"
                style={{ transformOrigin: "center bottom" }}
              >
                <img
                  src={assetUrl(url)}
                  alt={FRAME_LABELS[index]}
                  className="aspect-video w-full object-cover"
                />
                <figcaption className="border-t border-border px-4 py-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-accent">
                    {FRAME_LABELS[index]}
                  </p>
                  <p className="mt-0.5 font-mono text-xs text-muted">
                    align{" "}
                    {(
                      result.alignment_scores[
                        ["hook", "context", "delivery"][index] as keyof typeof result.alignment_scores
                      ] ?? 0
                    ).toFixed(3)}
                  </p>
                </figcaption>
              </motion.figure>
            ))}
          </motion.div>
        </div>
      </motion.div>

      <motion.section
        layout
        variants={reducedMotion ? undefined : tdsExpand}
        style={{ transformOrigin: "top center" }}
        className="rounded-xl border border-border bg-surface p-6"
      >
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-muted">
              Temporal Divergence Score (TDS)
            </h3>
            <motion.p
              className="mt-2 font-mono text-4xl font-bold text-foreground"
              initial={reducedMotion ? false : { scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={motionTransition(!!reducedMotion, SPRING_BOUNCY)}
            >
              {result.tds_score.toFixed(3)}
              <span className="ml-2 text-lg font-normal text-muted">/ 1.0</span>
            </motion.p>
          </div>
          <div className="h-2 w-full max-w-xs overflow-hidden rounded-full bg-surface-elevated sm:w-48">
            <motion.div
              className="h-full rounded-full bg-accent"
              initial={reducedMotion ? false : { width: 0 }}
              animate={{ width: `${Math.min(result.tds_score * 100, 100)}%` }}
              transition={motionTransition(!!reducedMotion, SPRING_EXPAND)}
            />
          </div>
        </div>
        <p className="mt-4 text-sm leading-relaxed text-muted">
          TDS measures visual change between the hook and delivery frames where higher means
          more temporal divergence between what the title promises and what the video shows.
        </p>
        {researchNote && (
          <div className="mt-4 rounded-lg border border-accent/20 bg-accent/5 p-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-accent">
              Study finding
            </p>
            <p className="mt-2 text-sm leading-relaxed text-foreground">{researchNote}</p>
          </div>
        )}
      </motion.section>

      <motion.section
        layout
        variants={reducedMotion ? undefined : panelReveal}
        className="rounded-xl border border-border bg-surface-elevated p-6"
      >
        <h3 className="text-sm font-semibold uppercase tracking-wider text-muted">
          Model explanation
        </h3>
        <p className="mt-3 text-base leading-relaxed">
          <WordReveal text={getDisplayExplanation(result)} startDelay={0.1} />
        </p>
      </motion.section>
    </motion.div>
  );
}

function HardCaseContent({ result }: { result: AnalysisResult }) {
  const textOnly = result.text_only!;
  const vtcfCorrect = result.vtcf_rescued ?? result.verdict === result.ground_truth;

  return (
    <>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-accent px-3 py-1 text-xs font-bold uppercase tracking-wider text-background">
          Judge highlight — hard case
        </span>
        {vtcfCorrect && (
          <span className="text-sm text-accent">VTCF rescues where title-only fails</span>
        )}
      </div>
      <p className="mb-6 text-sm text-muted">{result.example_description}</p>

      <div className="grid gap-4 md:grid-cols-2">
        <ComparisonCard
          label="BanglaBERT title-only"
          sublabel="Text baseline (wrong on this video)"
          verdict={textOnly.verdict}
          confidence={textOnly.confidence}
          wrong
        />
        <ComparisonCard
          label="Full VTCF (this demo)"
          sublabel="BanglaBERT + ViT fusion (correct)"
          verdict={result.verdict}
          confidence={result.confidence}
          wrong={false}
          highlight
        />
      </div>

      {result.ground_truth && (
        <p className="mt-4 text-center text-sm text-muted">
          Human ground truth:{" "}
          <span className="font-semibold text-foreground">{result.ground_truth}</span>
        </p>
      )}
    </>
  );
}

function ComparisonCard({
  label,
  sublabel,
  verdict,
  confidence,
  wrong,
  highlight,
}: {
  label: string;
  sublabel: string;
  verdict: string;
  confidence: number;
  wrong?: boolean;
  highlight?: boolean;
}) {
  const isClickbait = verdict === "CLICKBAIT";
  return (
    <div
      className={`rounded-xl border p-5 ${
        highlight ? "border-accent/50 bg-surface" : "border-border bg-background/50"
      }`}
    >
      <p className="text-xs font-semibold uppercase tracking-wider text-muted">{label}</p>
      <p className="text-xs text-muted">{sublabel}</p>
      <p
        className={`mt-3 font-mono text-2xl font-bold ${
          isClickbait ? "text-clickbait" : "text-genuine"
        }`}
      >
        {verdict}
      </p>
      <p className="mt-1 text-sm text-muted">{formatConfidence(confidence)} confidence</p>
      {wrong && (
        <p className="mt-2 text-xs font-semibold uppercase tracking-wider text-clickbait">
          ✗ Incorrect
        </p>
      )}
      {highlight && (
        <p className="mt-2 text-xs font-semibold uppercase tracking-wider text-genuine">
          ✓ Correct
        </p>
      )}
    </div>
  );
}
