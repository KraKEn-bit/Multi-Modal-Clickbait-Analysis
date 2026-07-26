"use client";

import { forwardRef, useState } from "react";
import { motion, useReducedMotion, type Variants } from "framer-motion";
import { ScrollReveal, ScrollRevealItem } from "@/components/ScrollReveal";
import { TdsExplainer } from "@/components/landing/TdsExplainer";
import { Brain, Eye, GitMerge, Lightbulb, type LucideIcon } from "lucide-react";

const HOVER_SPRING = { type: "spring", stiffness: 300, damping: 20 } as const;

const ICON_VARIANTS: Variants = {
  rest: { scale: 1, rotate: 0 },
  hover: { scale: 1.12, rotate: -6 },
};

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
  active,
  onActivate,
}: {
  icon: LucideIcon;
  title: string;
  body: string;
  step: string;
  active: boolean;
  onActivate: () => void;
}) {
  const reducedMotion = useReducedMotion();

  return (
    <motion.button
      type="button"
      onClick={onActivate}
      aria-pressed={active}
      initial="rest"
      whileHover={reducedMotion ? undefined : "hover"}
      animate="rest"
      variants={{ rest: { y: 0 }, hover: { y: -6 } }}
      transition={HOVER_SPRING}
      className={`science-pipeline-card soft-card group relative h-full w-full transform-gpu p-6 text-left will-change-transform ${
        active ? "!border-accent/40 ring-1 ring-accent/25" : ""
      }`}
    >
      <span className="data-point absolute right-5 top-5">{step}</span>

      <motion.span
        variants={reducedMotion ? undefined : ICON_VARIANTS}
        transition={HOVER_SPRING}
        className={`flex h-11 w-11 transform-gpu items-center justify-center rounded-xl transition-colors duration-200 will-change-transform ${
          active ? "bg-accent-soft" : "bg-surface-elevated group-hover:bg-accent-soft"
        }`}
      >
        <Icon
          className={`h-5 w-5 transition-colors duration-200 ${
            active ? "text-accent-strong" : "text-muted group-hover:text-accent-strong"
          }`}
          aria-hidden
        />
      </motion.span>

      <h3 className="mt-5 text-lg font-bold text-foreground">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>

      <span
        className={`mt-4 inline-flex items-center gap-1 text-xs font-semibold transition-colors duration-200 ${
          active ? "text-accent-strong" : "text-transparent group-hover:text-accent-strong/80"
        }`}
      >
        {active ? "Active layer" : "Inspect"}
        <span aria-hidden>→</span>
      </span>
    </motion.button>
  );
}

export const ScienceSection = forwardRef<HTMLElement>(function ScienceSection(_props, ref) {
  const [activeIndex, setActiveIndex] = useState(0);

  return (
    <ScrollReveal ref={ref} id="science" className="scroll-mt-8 py-20 sm:py-24">
      <div className="mx-auto max-w-6xl px-6">
        <ScrollRevealItem>
          <span className="eyebrow">The science</span>
          <h2 className="mt-3 max-w-3xl text-3xl font-semibold leading-snug tracking-tight text-foreground sm:text-4xl">
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
                active={activeIndex === index}
                onActivate={() => setActiveIndex(index)}
              />
            </ScrollRevealItem>
          ))}
        </div>

        <ScrollRevealItem delay={0.25}>
          <div className="soft-card mt-10 flex gap-4 border-accent/20 bg-accent-soft/40 p-6 sm:gap-5 sm:p-8">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-surface-elevated">
              <Lightbulb className="h-5 w-5 text-agent" aria-hidden />
            </span>
            <div>
              <span className="eyebrow">Distinctive finding</span>
              <p className="mt-2 max-w-3xl text-base leading-relaxed text-foreground">
                Clickbait videos show <span className="font-bold text-clickbait">less</span>{" "}
                visual change on average than genuine videos because clickbait often reuses
                static footage rather than bait-and-switching content.
              </p>
            </div>
          </div>
        </ScrollRevealItem>

        <TdsExplainer />
      </div>
    </ScrollReveal>
  );
});
