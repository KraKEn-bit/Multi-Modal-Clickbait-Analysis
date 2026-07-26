"use client";

import type { ReactNode } from "react";
import { ScrollRevealItem } from "@/components/ScrollReveal";
import { VideoBackdrop } from "@/components/VideoBackdrop";

type Props = {
  onSeeStudy: () => void;
  console: ReactNode;
};

/**
 * Dark agent hero. The console passed in is the real analysis tool —
 * paste a URL and the pipeline runs right here. Backdrop is a
 * <VideoBackdrop>: plug in a looping .webm for 3D visuals; without
 * one it renders a static CSS mesh gradient (zero canvas cost).
 */
export function HeroSection({ onSeeStudy, console: agentConsole }: Props) {
  return (
    <section id="hero" className="relative overflow-hidden scroll-mt-8">
      <VideoBackdrop />

      <div className="relative mx-auto max-w-4xl px-6 pb-20 pt-16 text-center sm:pt-24">
        <ScrollRevealItem>
          <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 font-mono text-[11px] tracking-wider text-agent">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-agent" aria-hidden />
            VTCF · AUTONOMOUS VIDEO AGENT
          </span>
        </ScrollRevealItem>

        <ScrollRevealItem delay={0.08}>
          <h1 className="mx-auto mt-7 max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight text-white sm:text-5xl lg:text-6xl">
            An agent that{" "}
            <span className="bg-gradient-to-r from-agent to-accent bg-clip-text text-transparent">
              watches the video
            </span>{" "}
            before you do.
          </h1>
        </ScrollRevealItem>

        <ScrollRevealItem delay={0.14}>
          <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-zinc-400 sm:text-lg">
            VTCF reads the title, samples the frames, and calls out Bangla
            YouTube clickbait when the footage won&apos;t deliver what the
            headline promises.
          </p>
        </ScrollRevealItem>

        {/* The console IS the product — live input, agent pipeline, verdict */}
        <ScrollRevealItem delay={0.2} className="mt-12">
          {agentConsole}
        </ScrollRevealItem>

        <ScrollRevealItem delay={0.26}>
          <button
            type="button"
            onClick={onSeeStudy}
            className="mt-8 font-mono text-xs text-muted transition-colors duration-200 hover:text-accent"
          >
            ↓ read the study behind the agent
          </button>
        </ScrollRevealItem>
      </div>
    </section>
  );
}
