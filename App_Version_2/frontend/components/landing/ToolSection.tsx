"use client";

import { forwardRef, type RefObject } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ExampleCards } from "@/components/ExampleCards";
import { ResultsView } from "@/components/ResultsView";
import { ScrollReveal } from "@/components/ScrollReveal";
import type { AnalysisResult } from "@/lib/types";
import { motionTransition, SPRING_GENTLE } from "@/lib/motion";

type Props = {
  onExampleSelect: (example: AnalysisResult) => void;
  loading: boolean;
  examples: AnalysisResult[];
  examplesError: string | null;
  researchNote: string;
  result: AnalysisResult | null;
  resultsRef: RefObject<HTMLElement | null>;
};

/**
 * Case files + full report. The live URL input lives in the hero
 * agent console; this section holds the pre-computed examples and
 * renders the full evidence report once a verdict lands.
 */
export const ToolSection = forwardRef<HTMLElement, Props>(function ToolSection(
  { onExampleSelect, loading, examples, examplesError, researchNote, result, resultsRef },
  ref,
) {
  const reducedMotion = useReducedMotion();

  return (
    <ScrollReveal ref={ref} id="tool" className="scroll-mt-8 py-16 sm:py-20">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mb-10 max-w-2xl">
          <span className="eyebrow">Case files</span>
          <h2 className="mt-2 text-3xl font-semibold leading-snug tracking-tight text-foreground sm:text-4xl">
            Open a case from the 805-video test set
          </h2>
          <p className="mt-3 text-sm leading-relaxed text-muted sm:text-base">
            Pre-computed results load instantly. Hard cases show where BanglaBERT
            title-only fails but full VTCF gets it right.
          </p>
        </div>

        {examplesError && (
          <p className="font-mono text-sm text-clickbait">{examplesError}</p>
        )}
        {examples.length > 0 && (
          <ExampleCards
            examples={examples}
            onSelect={onExampleSelect}
            disabled={loading}
          />
        )}

        <AnimatePresence initial={false}>
          {result && !loading && (
            <motion.section
              ref={resultsRef}
              key="analysis-results"
              id="analysis-results"
              initial={reducedMotion ? false : { opacity: 0, y: 56 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reducedMotion ? undefined : { opacity: 0, y: 32 }}
              transition={motionTransition(!!reducedMotion, SPRING_GENTLE)}
              className="mt-12 scroll-mt-8 transform-gpu"
            >
              <ResultsView result={result} researchNote={researchNote} />
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </ScrollReveal>
  );
});
