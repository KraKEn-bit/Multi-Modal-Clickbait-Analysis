"use client";

import { ScrollReveal, ScrollRevealItem } from "@/components/ScrollReveal";
import { CountUp } from "@/components/CountUp";
import { Check } from "lucide-react";

const VALUE_POINTS = [
  "Reads the title's promise with a BanglaBERT text encoder",
  "Watches hook, context and delivery frames with a Vision Transformer",
  "Cross-attends text against visuals to catch the bait-and-switch",
] as const;

const STATS = [
  {
    value: 8047,
    suffix: "",
    label: "Videos studied",
    detail: "Human-labeled Bangla YouTube videos in the research corpus",
  },
  {
    value: 160,
    suffix: "",
    label: "Test videos",
    detail: "Held-out evaluation set behind the headline metrics",
  },
] as const;

export function ValueSection() {
  return (
    <ScrollReveal id="value" className="scroll-mt-8 py-20 sm:py-24">
      <div className="mx-auto max-w-6xl px-6">
        <div className="grid items-center gap-12 lg:grid-cols-2 lg:gap-16">
          {/* Value proposition */}
          <div>
            <ScrollRevealItem>
              <span className="eyebrow">Why VTCF</span>
              <h2 className="mt-4 text-3xl font-semibold leading-snug tracking-tight text-foreground sm:text-4xl">
                Titles can lie.
                <br />
                Frames can&apos;t hide.
              </h2>
            </ScrollRevealItem>

            <ScrollRevealItem delay={0.08}>
              <p className="mt-5 max-w-lg text-base leading-relaxed text-muted">
                Text-only classifiers miss clickbait when the headline reads like
                legitimate news. VTCF pairs language understanding with visual
                evidence, so the verdict reflects what the video actually shows —
                not just what it claims.
              </p>
            </ScrollRevealItem>

            <ScrollRevealItem delay={0.14}>
              <ul className="mt-7 space-y-3.5">
                {VALUE_POINTS.map((point) => (
                  <li key={point} className="flex items-start gap-3">
                    <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-soft">
                      <Check className="h-3 w-3 text-accent-strong" aria-hidden />
                    </span>
                    <span className="text-sm leading-relaxed text-foreground/85 sm:text-base">
                      {point}
                    </span>
                  </li>
                ))}
              </ul>
            </ScrollRevealItem>
          </div>

          {/* Big numbers — Anime.js count-up on scroll */}
          <ScrollRevealItem delay={0.12}>
            <div className="grid gap-4">
              {STATS.map((stat) => (
                <div
                  key={stat.label}
                  className="soft-card soft-card--hover flex items-center gap-6 p-6 sm:p-7"
                >
                  <CountUp
                    to={stat.value}
                    suffix={stat.suffix}
                    className="min-w-[7rem] text-4xl font-bold tracking-tight text-accent-strong sm:text-5xl"
                  />
                  <div>
                    <p className="text-sm font-bold text-foreground">{stat.label}</p>
                    <p className="mt-1 text-xs leading-relaxed text-muted">{stat.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </ScrollRevealItem>
        </div>
      </div>
    </ScrollReveal>
  );
}
