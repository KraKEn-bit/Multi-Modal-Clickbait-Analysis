"use client";

import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowRight, HelpCircle, Layers, Scale } from "lucide-react";
import { ScrollRevealItem } from "@/components/ScrollReveal";
import { motionTransition, SPRING_GENTLE, SPRING_SNAPPY } from "@/lib/motion";
import { FRAME_LABELS } from "@/lib/types";

/** Plain-language bands — conceptual only, not classification thresholds. */
const TDS_BANDS = [
  {
    min: 0,
    max: 0.33,
    label: "Mostly steady",
    detail: "Visuals stay broadly similar from the opening hook through to delivery.",
  },
  {
    min: 0.33,
    max: 0.66,
    label: "Moderate shift",
    detail: "Noticeable visual change across the three sampled moments in the timeline.",
  },
  {
    min: 0.66,
    max: 1,
    label: "Strong shift",
    detail: "Substantial visual evolution between the hook and the delivery frame.",
  },
] as const;

const FAQ_ITEMS = [
  {
    id: "what",
    icon: HelpCircle,
    question: "What is TDS?",
    answer:
      "Temporal Divergence Score (TDS) is a number from 0 to 1 that summarizes how much a video's visuals change across three moments : hook, context, and delivery. It gives you a quick read on temporal visual change, not a verdict by itself.",
  },
  {
    id: "read",
    icon: Scale,
    question: "How should I read it?",
    answer:
      "Use TDS alongside the verdict, the three frame thumbnails, and the written explanation. A higher score means more visual shift over time; a lower score means steadier footage. Context always matters.",
  },
  {
    id: "not",
    icon: Layers,
    question: "What it is not",
    answer:
      "TDS is not a standalone clickbait detector and does not replace the full multimodal model. It does not judge title quality alone, it describes visual change across the sampled frames only.",
  },
] as const;

function bandForScore(score: number) {
  if (score < 0.34) return TDS_BANDS[0];
  if (score < 0.67) return TDS_BANDS[1];
  return TDS_BANDS[2];
}

export function TdsExplainer() {
  const reducedMotion = useReducedMotion();
  const [score, setScore] = useState(0.52);
  const [openFaq, setOpenFaq] = useState<string>("what");
  const band = bandForScore(score);

  return (
    <ScrollRevealItem delay={0.32} className="mt-10">
      <div className="rounded-2xl border border-border bg-surface p-6 sm:p-8">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-accent">
          Score guide
        </p>
        <h3 className="mt-2 font-[family-name:var(--font-fraunces)] text-xl font-bold tracking-tight sm:text-2xl">
          What is Temporal Divergence Score (TDS)?
        </h3>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted sm:text-base">
          A compact signal for how much a video&apos;s visuals move across three checkpoints —
          always shown next to the frames and explanation in your results.
        </p>

        {/* Interactive slider demo */}
        <div className="mt-8 rounded-xl border border-border bg-background/60 p-5 sm:p-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted">
                Explore a sample score
              </p>
              <p className="mt-1 font-mono text-3xl font-bold text-foreground">
                {score.toFixed(2)}
                <span className="ml-2 text-base font-normal text-muted">/ 1.0</span>
              </p>
            </div>
            <AnimatePresence mode="wait">
              <motion.p
                key={band.label}
                initial={reducedMotion ? false : { opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reducedMotion ? undefined : { opacity: 0, y: -4 }}
                transition={motionTransition(!!reducedMotion, SPRING_SNAPPY)}
                className="rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-accent"
              >
                {band.label}
              </motion.p>
            </AnimatePresence>
          </div>

          <div className="mt-5">
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(score * 100)}
              onChange={(e) => setScore(Number(e.target.value) / 100)}
              aria-label="Explore sample Temporal Divergence Score"
              className="tds-slider w-full cursor-pointer accent-accent"
            />
            <div className="mt-2 flex justify-between text-[10px] uppercase tracking-wider text-muted">
              <span>0 — steady</span>
              <span>1 — strong shift</span>
            </div>
          </div>

          <motion.div
            className="mt-4 h-2 overflow-hidden rounded-full bg-surface-elevated"
            layout
          >
            <motion.div
              className="h-full rounded-full bg-accent"
              animate={{ width: `${score * 100}%` }}
              transition={motionTransition(!!reducedMotion, SPRING_GENTLE)}
            />
          </motion.div>

          <AnimatePresence mode="wait">
            <motion.p
              key={band.detail}
              initial={reducedMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reducedMotion ? undefined : { opacity: 0 }}
              transition={motionTransition(!!reducedMotion, SPRING_GENTLE)}
              className="mt-4 text-sm leading-relaxed text-muted"
            >
              {band.detail}
            </motion.p>
          </AnimatePresence>

          {/* Timeline nodes — the three moments, no selection algorithm exposed */}
          <div className="mt-6 flex items-center justify-between gap-2">
            {FRAME_LABELS.map((label, index) => (
              <div key={label} className="flex flex-1 flex-col items-center gap-2">
                <motion.span
                  className="flex h-8 w-8 items-center justify-center rounded-full border border-border bg-surface text-[10px] font-bold uppercase text-accent"
                  animate={{
                    scale: score > index * 0.28 ? 1.05 : 1,
                    borderColor:
                      score > index * 0.28
                        ? "rgba(0, 210, 255, 0.55)"
                        : "rgba(42, 52, 48, 1)",
                  }}
                  transition={motionTransition(!!reducedMotion, SPRING_SNAPPY)}
                >
                  {index + 1}
                </motion.span>
                <span className="text-center text-[10px] font-semibold uppercase tracking-wider text-muted">
                  {label}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-4 flex items-center gap-1.5 text-xs text-muted">
            <ArrowRight className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden />
            VTCF samples these three moments to summarize visual change over time.
          </p>
        </div>

        {/* FAQ accordion */}
        <div className="mt-6 divide-y divide-border rounded-xl border border-border">
          {FAQ_ITEMS.map(({ id, icon: Icon, question, answer }) => {
            const isOpen = openFaq === id;
            return (
              <div key={id}>
                <button
                  type="button"
                  onClick={() => setOpenFaq(isOpen ? "" : id)}
                  className="flex w-full items-center gap-3 px-4 py-4 text-left transition hover:bg-surface-elevated/50 sm:px-5"
                  aria-expanded={isOpen}
                >
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-border bg-background">
                    <Icon className="h-4 w-4 text-accent" aria-hidden />
                  </span>
                  <span className="flex-1 text-sm font-semibold text-foreground">{question}</span>
                  <motion.span
                    animate={{ rotate: isOpen ? 45 : 0 }}
                    transition={motionTransition(!!reducedMotion, SPRING_SNAPPY)}
                    className="text-lg leading-none text-muted"
                    aria-hidden
                  >
                    +
                  </motion.span>
                </button>
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      initial={reducedMotion ? false : { height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={reducedMotion ? undefined : { height: 0, opacity: 0 }}
                      transition={motionTransition(!!reducedMotion, SPRING_GENTLE)}
                      className="overflow-hidden"
                    >
                      <p className="px-4 pb-4 pl-[3.25rem] text-sm leading-relaxed text-muted sm:px-5 sm:pl-[4.5rem]">
                        {answer}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </div>
      </div>
    </ScrollRevealItem>
  );
}
