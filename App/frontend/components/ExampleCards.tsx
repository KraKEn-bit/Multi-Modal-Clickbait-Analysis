"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { SearchCheck } from "lucide-react";
import type { AnalysisResult } from "@/lib/types";
import { youtubeThumb } from "@/lib/api";
import {
  motionTransition,
  SPRING_GENTLE,
  SPRING_SNAPPY,
  usePrefersReducedMotion,
} from "@/lib/motion";

type Props = {
  examples: AnalysisResult[];
  onSelect: (example: AnalysisResult) => void;
  disabled?: boolean;
};

const CATEGORY_STYLES: Record<string, string> = {
  clickbait: "border-clickbait/40 text-clickbait",
  genuine: "border-genuine/40 text-genuine",
  hard: "border-accent/50 text-accent",
};

function ExampleCard({
  example,
  disabled,
  onSelect,
}: {
  example: AnalysisResult;
  disabled?: boolean;
  onSelect: (example: AnalysisResult) => void;
}) {
  const reducedMotion = usePrefersReducedMotion();
  const [hovered, setHovered] = useState(false);
  const showOverlay = hovered && !reducedMotion && !disabled;

  return (
    <motion.button
      type="button"
      disabled={disabled}
      onClick={() => onSelect(example)}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      whileHover={
        reducedMotion || disabled
          ? undefined
          : {
              scale: 1.02,
              y: -6,
              boxShadow:
                "0 16px 48px rgba(0, 210, 255, 0.28), 0 0 0 1px rgba(0, 210, 255, 0.5)",
            }
      }
      whileTap={reducedMotion || disabled ? undefined : { scale: 0.985 }}
      transition={motionTransition(reducedMotion, SPRING_GENTLE)}
      className="group relative overflow-hidden rounded-xl border border-border bg-surface p-4 text-left disabled:opacity-50"
      style={{ boxShadow: "0 0 0 rgba(0, 210, 255, 0)" }}
    >
      <div className="flex gap-4">
        <div className="relative h-20 w-32 shrink-0 overflow-hidden rounded-lg">
          <motion.img
            src={youtubeThumb(example.video_id)}
            alt=""
            className="h-full w-full object-cover"
            animate={showOverlay ? { scale: 1.06 } : { scale: 1 }}
            transition={motionTransition(reducedMotion, SPRING_GENTLE)}
          />

          <AnimatePresence>
            {showOverlay && (
              <>
                <motion.div
                  key="evidence-gradient"
                  initial={{ y: "100%" }}
                  animate={{ y: "0%" }}
                  exit={{ y: "100%" }}
                  transition={motionTransition(false, SPRING_SNAPPY)}
                  className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/45 to-transparent"
                />
                <motion.div
                  key="evidence-label"
                  initial={{ y: "100%", opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  exit={{ y: "100%", opacity: 0 }}
                  transition={motionTransition(false, {
                    ...SPRING_SNAPPY,
                    delay: 0.05,
                  })}
                  className="absolute inset-x-0 bottom-0 flex items-center justify-center gap-1.5 pb-2.5 text-[11px] font-semibold uppercase tracking-wider text-[#E8E6DF]"
                >
                  <SearchCheck className="h-3.5 w-3.5 text-accent" aria-hidden />
                  Review Evidence
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
                CATEGORY_STYLES[example.category ?? "hard"]
              }`}
            >
              {example.example_label ?? example.category}
            </span>
            {example.vtcf_rescued && (
              <span className="rounded-full bg-accent/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent">
                VTCF rescue
              </span>
            )}
          </div>
          <p className="line-clamp-2 text-sm font-medium leading-snug text-foreground">
            {example.title}
          </p>
          <p className="mt-1 text-xs text-muted">
            {example.example_description ?? "Pre-computed — loads instantly"}
          </p>
        </div>
      </div>
    </motion.button>
  );
}

export function ExampleCards({ examples, onSelect, disabled }: Props) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {examples.map((example) => (
        <ExampleCard
          key={example.video_id}
          example={example}
          onSelect={onSelect}
          disabled={disabled}
        />
      ))}
    </div>
  );
}
