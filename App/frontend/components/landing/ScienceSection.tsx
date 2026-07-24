"use client";

import { forwardRef, useState } from "react";
import { motion } from "framer-motion";
import { ScrollReveal, ScrollRevealItem } from "@/components/ScrollReveal";
import { TdsExplainer } from "@/components/landing/TdsExplainer";
import { motionTransition, SPRING_GENTLE, SPRING_SNAPPY, usePrefersReducedMotion } from "@/lib/motion";
import { Brain, Eye, GitMerge, type LucideIcon } from "lucide-react";

const PIPELINE_CARDS: Array<{
  icon: LucideIcon;
  title: string;
  body: string;
  step: string;
}> = [
  {
    step: "01",
    icon: Brain,
    title: "Text Analysis",
    body: "BanglaBERT reads the title's promise",
  },
  {
    step: "02",
    icon: Eye,
    title: "Visual Analysis",
    body: "A Vision Transformer (ViT) reads hook, context, and delivery frames",
  },
  {
    step: "03",
    icon: GitMerge,
    title: "Cross-Modal Fusion",
    body: "Cross-attention compares title against visuals to catch what text alone misses",
  },
];

function PipelineCard({
  icon: Icon,
  title,
  body,
  step,
  index,
  active,
  onActivate,
  reducedMotion,
}: {
  icon: LucideIcon;
  title: string;
  body: string;
  step: string;
  index: number;
  active: boolean;
  onActivate: () => void;
  reducedMotion: boolean;
}) {
  return (
    <motion.button
      type="button"
      onClick={onActivate}
      aria-pressed={active}
      whileHover={
        reducedMotion
          ? undefined
          : {
              y: -6,
              borderColor: "rgba(0, 210, 255, 0.45)",
              boxShadow: "0 16px 48px rgba(0, 210, 255, 0.15)",
            }
      }
      whileTap={reducedMotion ? undefined : { scale: 0.985 }}
      animate={
        active && !reducedMotion
          ? {
              borderColor: "rgba(0, 210, 255, 0.55)",
              boxShadow: "0 0 0 1px rgba(0, 210, 255, 0.25), 0 20px 50px rgba(0, 210, 255, 0.18)",
            }
          : {}
      }
      transition={motionTransition(reducedMotion, SPRING_GENTLE)}
      className={`science-pipeline-card group relative h-full w-full rounded-2xl border p-6 text-left transition-colors ${
        active ? "border-accent/50 bg-surface-elevated/80" : "border-border bg-surface"
      }`}
    >
      <span className="absolute right-5 top-5 font-mono text-[10px] tracking-widest text-muted/60">
        {step}
      </span>

      <motion.span
        className={`flex h-11 w-11 items-center justify-center rounded-xl border transition-colors ${
          active
            ? "border-accent/40 bg-accent/15"
            : "border-border bg-background group-hover:border-accent/30 group-hover:bg-accent/10"
        }`}
        animate={active && !reducedMotion ? { scale: [1, 1.08, 1] } : { scale: 1 }}
        transition={
          active && !reducedMotion
            ? { duration: 2, repeat: Infinity, ease: "easeInOut" }
            : motionTransition(reducedMotion, SPRING_SNAPPY)
        }
      >
        <Icon
          className={`h-5 w-5 transition-colors ${
            active ? "text-accent" : "text-accent/80 group-hover:text-accent"
          }`}
          aria-hidden
        />
      </motion.span>

      <h3 className="mt-5 text-lg font-semibold text-foreground">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>

      <span
        className={`mt-4 inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider transition-colors ${
          active ? "text-accent" : "text-accent/0 group-hover:text-accent/80"
        }`}
      >
        {active ? "Active layer" : "Inspect"}
        <span aria-hidden>→</span>
      </span>

      {index < 2 && (
        <span
          className="pointer-events-none absolute -right-3 top-1/2 hidden h-px w-6 bg-gradient-to-r from-accent/40 to-transparent md:block"
          aria-hidden
        />
      )}
    </motion.button>
  );
}

export const ScienceSection = forwardRef<HTMLElement>(function ScienceSection(_props, ref) {
  const reducedMotion = usePrefersReducedMotion();
  const [activeIndex, setActiveIndex] = useState(0);

  return (
    <ScrollReveal
      ref={ref}
      id="science"
      className="scroll-mt-8 border-t border-border/50 py-16 sm:py-24"
    >
      <div className="mx-auto max-w-6xl">
        <ScrollRevealItem>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-accent">
            The science
          </p>
          <h2 className="mt-3 max-w-3xl font-[family-name:var(--font-fraunces)] text-2xl font-bold tracking-tight sm:text-4xl">
            Powered by VTCF : Visual-Temporal Contradiction Framework
          </h2>
        </ScrollRevealItem>

        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {PIPELINE_CARDS.map(({ icon, title, body, step }, index) => (
            <ScrollRevealItem key={title} delay={0.08 + index * 0.1}>
              <PipelineCard
                icon={icon}
                title={title}
                body={body}
                step={step}
                index={index}
                active={activeIndex === index}
                onActivate={() => setActiveIndex(index)}
                reducedMotion={reducedMotion}
              />
            </ScrollRevealItem>
          ))}
        </div>

        <ScrollRevealItem delay={0.25}>
          <motion.div
            whileHover={
              reducedMotion
                ? undefined
                : { borderColor: "rgba(0, 210, 255, 0.35)", y: -2 }
            }
            transition={motionTransition(reducedMotion, SPRING_GENTLE)}
            className="mt-10 rounded-2xl border border-accent/25 bg-accent/5 p-6 sm:p-8"
          >
            <p className="text-xs font-semibold uppercase tracking-wider text-accent">
              Distinctive finding
            </p>
            <p className="mt-3 max-w-3xl text-base leading-relaxed text-foreground">
              Clickbait videos show <span className="font-semibold text-clickbait">less</span>{" "}
              visual change on average than genuine videos because clickbait often reuses static
              footage rather than bait-and-switching content.
            </p>
          </motion.div>
        </ScrollRevealItem>

        <TdsExplainer />
      </div>
    </ScrollReveal>
  );
});
