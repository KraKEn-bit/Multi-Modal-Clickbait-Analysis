"use client";

import { motion } from "framer-motion";
import { ScrollReveal, ScrollRevealItem } from "@/components/ScrollReveal";
import { motionTransition, SPRING_GENTLE, usePrefersReducedMotion } from "@/lib/motion";
import { AlertTriangle, Eye, FileText } from "lucide-react";

/** Why text-only fails — distinct from the Science pipeline cards. */
const FAILURE_MODES = [
  {
    icon: FileText,
    title: "Polished headlines",
    detail:
      "A title can read like legitimate news while the footage tells a different story entirely.",
  },
  {
    icon: Eye,
    title: "Visual bait-and-switch",
    detail:
      "Sensational thumbnails and opening hooks often never appear again by the delivery frame.",
  },
  {
    icon: AlertTriangle,
    title: "Title-only blind spots",
    detail:
      "Text classifiers never see the frames — so they miss contradiction when wording alone looks fine.",
  },
] as const;

export function ProblemSection() {
  const reducedMotion = usePrefersReducedMotion();

  return (
    <ScrollReveal
      id="problem"
      className="scroll-mt-8 border-t border-border/50 py-16 sm:py-20"
    >
      <div className="mx-auto max-w-6xl">
        <ScrollRevealItem>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted">
            The problem
          </p>
          <h2 className="mt-3 max-w-3xl font-[family-name:var(--font-fraunces)] text-2xl font-bold tracking-tight sm:text-4xl">
            Titles promise. Videos don&apos;t always deliver.
          </h2>
        </ScrollRevealItem>

        <ScrollRevealItem delay={0.1} className="mt-6 max-w-3xl">
          <p className="text-base leading-relaxed text-muted sm:text-lg">
            Bangla YouTube clickbait often pairs sensational headlines with footage that never
            delivers on the promise, a classic bait-and-switch. Text-only classifiers can miss
            these cases entirely when the title itself reads well: polished wording, plausible
            news framing, or emotionally charged phrasing that looks genuine on paper. You need
            to see what the video actually shows.
          </p>
        </ScrollRevealItem>

        <div className="mt-12 grid gap-4 sm:grid-cols-3">
          {FAILURE_MODES.map(({ icon: Icon, title, detail }, index) => (
            <ScrollRevealItem key={title} delay={0.08 + index * 0.08}>
              <motion.div
                whileHover={
                  reducedMotion
                    ? undefined
                    : {
                        y: -4,
                        borderColor: "rgba(0, 210, 255, 0.35)",
                        boxShadow: "0 12px 40px rgba(0, 210, 255, 0.12)",
                      }
                }
                transition={motionTransition(reducedMotion, SPRING_GENTLE)}
                className="h-full rounded-2xl border border-border bg-surface p-6"
              >
                <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-accent/25 bg-accent/10">
                  <Icon className="h-5 w-5 text-accent" aria-hidden />
                </span>
                <p className="mt-4 text-sm font-semibold text-foreground">{title}</p>
                <p className="mt-2 text-sm leading-relaxed text-muted">{detail}</p>
              </motion.div>
            </ScrollRevealItem>
          ))}
        </div>

        <ScrollRevealItem delay={0.28}>
          <div className="mt-10 grid gap-4 border-t border-border/50 pt-10 sm:grid-cols-3">
            {[
              {
                value: "99.63%",
                label: "F1 score",
                detail: "Full VTCF model on the main test set (8,047 videos)",
              },
              {
                value: "100% vs 64%",
                label: "Hard-case rescue",
                detail:
                  "VTCF vs. speech/summary detection on 33 hardest title-only failures",
              },
              {
                value: "8,047",
                label: "Videos analyzed",
                detail: "Human-labeled Bangla YouTube videos in our evaluation set",
              },
            ].map((stat) => (
              <div
                key={stat.label}
                className="rounded-xl border border-border/60 bg-surface-elevated/40 px-5 py-4"
              >
                <p className="font-mono text-2xl font-bold text-accent sm:text-3xl">
                  {stat.value}
                </p>
                <p className="mt-1 text-xs font-semibold uppercase tracking-wider text-foreground">
                  {stat.label}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-muted">{stat.detail}</p>
              </div>
            ))}
          </div>
        </ScrollRevealItem>
      </div>
    </ScrollReveal>
  );
}
