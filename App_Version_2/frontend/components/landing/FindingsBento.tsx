"use client";

import { motion, useReducedMotion, type Variants } from "framer-motion";
import { ScrollReveal, ScrollRevealItem } from "@/components/ScrollReveal";
import { TdsDivergenceChart } from "@/components/charts/TdsDivergenceChart";
import { ScoreRings } from "@/components/charts/ScoreRings";
import { Layers, LifeBuoy, ShieldAlert, type LucideIcon } from "lucide-react";

/**
 * Research findings — chart modules + kokonut spring-hover cards.
 * Background aesthetics come from the global CSS mesh (no canvas).
 */

const HOVER_SPRING = { type: "spring", stiffness: 300, damping: 20 } as const;

const ICON_VARIANTS: Variants = {
  rest: { scale: 1, rotate: 0 },
  hover: { scale: 1.12, rotate: 6 },
};

function FindingCard({
  icon: Icon,
  iconClass,
  iconBoxClass,
  kicker,
  title,
  children,
}: {
  icon: LucideIcon;
  iconClass: string;
  iconBoxClass: string;
  kicker: string;
  title: string;
  children: React.ReactNode;
}) {
  const reducedMotion = useReducedMotion();

  return (
    <motion.div
      initial="rest"
      whileHover={reducedMotion ? undefined : "hover"}
      animate="rest"
      variants={{ rest: { y: 0 }, hover: { y: -6 } }}
      transition={HOVER_SPRING}
      className="soft-card h-full transform-gpu p-7 will-change-transform"
    >
      <motion.span
        variants={reducedMotion ? undefined : ICON_VARIANTS}
        transition={HOVER_SPRING}
        className={`flex h-11 w-11 transform-gpu items-center justify-center rounded-xl will-change-transform ${iconBoxClass}`}
      >
        <Icon className={`h-5 w-5 ${iconClass}`} aria-hidden />
      </motion.span>
      <p className="mt-5 text-xs font-semibold uppercase tracking-wider text-muted">{kicker}</p>
      <h3 className="mt-1.5 text-lg font-bold text-foreground">{title}</h3>
      <div className="mt-2.5 text-sm leading-relaxed text-muted">{children}</div>
    </motion.div>
  );
}

export function FindingsBento() {
  return (
    <ScrollReveal
      id="findings"
      className="relative scroll-mt-8 py-20 sm:py-24"
    >
      <div className="relative mx-auto max-w-6xl px-6">
        <ScrollRevealItem>
          <span className="eyebrow">Research findings</span>
          <h2 className="mt-4 max-w-2xl text-3xl font-semibold leading-snug tracking-tight text-foreground sm:text-4xl">
            What the hard cases taught us
          </h2>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-muted">
            Across the 160-video test set, 9 videos fooled every text channel at
            once. The data tells the story better than we can.
          </p>
        </ScrollRevealItem>

        {/* Phase 2 — dual chart cards */}
        <div className="mt-12 grid gap-5 lg:grid-cols-5">
          <ScrollRevealItem delay={0.06} className="lg:col-span-3">
            <div className="soft-card h-full p-7">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted">
                Temporal divergence
              </p>
              <h3 className="mt-1.5 text-lg font-bold text-foreground">
                Genuine videos move. Clickbait stalls.
              </h3>
              <div className="mt-6">
                <TdsDivergenceChart />
              </div>
              <p className="mt-4 text-xs leading-relaxed text-muted">
                Mean temporal divergence accumulated across the video timeline.
                Endpoint means are measured study values; the curve shape between
                checkpoints is illustrative.
              </p>
            </div>
          </ScrollRevealItem>

          <ScrollRevealItem delay={0.12} className="lg:col-span-2">
            <div className="soft-card flex h-full flex-col justify-center p-7">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted">
                Headline metrics
              </p>
              <h3 className="mt-1.5 text-lg font-bold text-foreground">
                Accurate on the easy cases. Unbeaten on the hard ones.
              </h3>
              <div className="mt-8">
                <ScoreRings />
              </div>
            </div>
          </ScrollRevealItem>
        </div>

        {/* Phase 3 — kokonut-style finding cards */}
        <div className="mt-5 grid gap-5 md:grid-cols-3">
          <ScrollRevealItem delay={0.08}>
            <FindingCard
              icon={Layers}
              iconClass="text-accent-strong"
              iconBoxClass="bg-accent-soft"
              kicker="Finding 1"
              title="Visual-Temporal Contradiction"
            >
              Clickbait changes its visuals far less than genuine content — bait
              recycles static footage instead of telling a story. That gap
              (TDS 0.383 vs 0.644) is the detection signal.
            </FindingCard>
          </ScrollRevealItem>

          <ScrollRevealItem delay={0.14}>
            <FindingCard
              icon={ShieldAlert}
              iconClass="text-warning"
              iconBoxClass="bg-warning/10"
              kicker="Finding 2"
              title="Production-Style Deception"
            >
              Polished clickbait fools every text channel at once: plausible
              speech, clean on-screen text. When all the words lie consistently,
              only the frames can disagree.
            </FindingCard>
          </ScrollRevealItem>

          <ScrollRevealItem delay={0.2}>
            <FindingCard
              icon={LifeBuoy}
              iconClass="text-accent-strong"
              iconBoxClass="bg-accent-soft"
              kicker="The rescue"
              title="9 text-blind cases, 9 recovered"
            >
              Every hard case that defeated the text channels was correctly
              classified once visual evidence joined the decision — versus 64%
              for speech and summary baselines.
            </FindingCard>
          </ScrollRevealItem>
        </div>
      </div>
    </ScrollReveal>
  );
}
