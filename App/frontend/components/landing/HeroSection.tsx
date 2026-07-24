"use client";

import dynamic from "next/dynamic";
import { useRef } from "react";
import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { ScrollRevealItem } from "@/components/ScrollReveal";
import { usePrefersReducedMotion } from "@/lib/motion";

const HeroSplineCanvas = dynamic(
  () => import("@/components/HeroSplineCanvas").then((mod) => mod.HeroSplineCanvas),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full w-full items-center justify-center bg-[#080C14]">
        <div className="h-8 w-8 animate-pulse rounded-full border border-accent/30 bg-accent/10" />
      </div>
    ),
  },
);

type Props = {
  onStartAnalyzing: () => void;
  onSeeStudy: () => void;
};

export function HeroSection({ onStartAnalyzing, onSeeStudy }: Props) {
  const heroRef = useRef<HTMLElement>(null);
  const reducedMotion = usePrefersReducedMotion();
  const framerReduced = useReducedMotion();

  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });

  const canvasOpacity = useTransform(
    scrollYProgress,
    [0, 0.55, 0.9],
    framerReduced ? [0.7, 0.7, 0.7] : [0.7, 0.35, 0],
  );
  const canvasScale = useTransform(
    scrollYProgress,
    [0, 0.9],
    framerReduced ? [1, 1] : [1, 0.88],
  );
  const canvasY = useTransform(
    scrollYProgress,
    [0, 1],
    framerReduced ? [0, 0] : [0, 48],
  );

  return (
    <section
      ref={heroRef}
      id="hero"
      className="relative min-h-screen scroll-mt-8 overflow-hidden py-16 sm:py-24 lg:py-28"
    >
      {/* Spline WebGL background */}
      <motion.div
        className="absolute inset-0 z-0 opacity-70"
        style={{
          opacity: canvasOpacity,
          scale: canvasScale,
          y: canvasY,
        }}
        aria-hidden="true"
      >
        {!reducedMotion && (
          <div className="hero-spline-filter absolute inset-0 z-0 h-full min-h-screen w-full overflow-hidden">
            <HeroSplineCanvas />
          </div>
        )}

        <div
          className="pointer-events-none absolute inset-0 z-0 bg-[radial-gradient(ellipse_at_center,_transparent_40%,_#080C14_100%)]"
          aria-hidden="true"
        />
      </motion.div>

      {/* Foreground copy */}
      <div className="relative z-10 mx-auto max-w-4xl px-0 pointer-events-none">
        <ScrollRevealItem>
          <div className="hero-glass mb-5 inline-block rounded-xl px-4 py-2">
            <p className="hero-eyebrow text-xs font-semibold uppercase">
              Visual-Temporal Contradiction Framework
            </p>
          </div>
        </ScrollRevealItem>

        <ScrollRevealItem delay={0.08}>
          <div className="hero-glass rounded-2xl p-6 sm:p-8">
            <h1 className="hero-headline text-[2.125rem] leading-[1.12] tracking-[-0.02em] sm:text-5xl sm:leading-[1.08] lg:text-[3.25rem]">
              Detecting the clickbait hiding behind polished titles.
            </h1>
          </div>
        </ScrollRevealItem>

        <ScrollRevealItem delay={0.14}>
          <div className="hero-glass mt-5 max-w-2xl rounded-2xl p-5 sm:p-6">
            <p className="hero-body text-lg leading-relaxed sm:text-xl">
              VTCF watches what a video actually shows, not just what its title promises to
              catch bait-and-switch content that text-only models miss.
            </p>
          </div>
        </ScrollRevealItem>

        <ScrollRevealItem delay={0.2}>
          <div className="pointer-events-auto mt-10 flex flex-col gap-3 sm:flex-row sm:items-center">
            <button
              type="button"
              onClick={onStartAnalyzing}
              className="rounded-xl bg-accent px-7 py-3.5 text-sm font-semibold text-background transition hover:bg-accent/90"
            >
              Start Analyzing
            </button>
            <button
              type="button"
              onClick={onSeeStudy}
              className="hero-glass rounded-xl px-7 py-3.5 text-sm font-semibold text-foreground transition hover:border-accent/40 hover:text-accent"
            >
              See the Study
            </button>
          </div>
        </ScrollRevealItem>
      </div>
    </section>
  );
}
