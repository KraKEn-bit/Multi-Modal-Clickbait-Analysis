"use client";

import { forwardRef, type RefObject } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { AnalyzeLivePanel } from "@/components/landing/AnalyzeLivePanel";
import { ExampleCards } from "@/components/ExampleCards";
import { PipelineProgress } from "@/components/PipelineProgress";
import { ResultsView } from "@/components/ResultsView";
import { ScrollReveal } from "@/components/ScrollReveal";
import type { AnalysisResult, AnalyzeEstimate } from "@/lib/types";
import { motionTransition, SPRING_GENTLE } from "@/lib/motion";

type Props = {
  url: string;
  onUrlChange: (url: string) => void;
  onAnalyze: () => void;
  onExampleSelect: (example: AnalysisResult) => void;
  loading: boolean;
  loadComplete: boolean;
  error: string | null;
  examples: AnalysisResult[];
  examplesError: string | null;
  researchNote: string;
  result: AnalysisResult | null;
  timeEstimate: AnalyzeEstimate | null;
  stageIndex: number;
  resultsRef: RefObject<HTMLElement | null>;
};

export const ToolSection = forwardRef<HTMLElement, Props>(function ToolSection(
  {
    url,
    onUrlChange,
    onAnalyze,
    onExampleSelect,
    loading,
    loadComplete,
    error,
    examples,
    examplesError,
    researchNote,
    result,
    timeEstimate,
    stageIndex,
    resultsRef,
  },
  ref,
) {
  const reducedMotion = useReducedMotion();
  const showResultsPanel = loading || !!result;

  return (
    <ScrollReveal
      ref={ref}
      id="tool"
      className="scroll-mt-8 border-t border-border/50 py-16 sm:py-20"
    >
      <div className="mx-auto max-w-6xl">
        <div className="mb-10 max-w-2xl">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-accent">
            Live analysis
          </p>
          <h2 className="mt-2 font-[family-name:var(--font-fraunces)] text-2xl font-bold tracking-tight sm:text-3xl">
            Paste a URL or open a case file
          </h2>
          <p className="mt-3 text-sm leading-relaxed text-muted sm:text-base">
            Run VTCF on any Bangla YouTube video, or load a pre-computed example to see how
            visual evidence changes the verdict.
          </p>
        </div>

        <AnalyzeLivePanel
          url={url}
          onUrlChange={onUrlChange}
          onAnalyze={onAnalyze}
          loading={loading}
          error={error}
        />

        <div className="mb-10 mt-10">
          <h3 className="mb-1 text-lg font-semibold">Example videos</h3>
          <p className="mb-4 text-sm text-muted">
            Pre-computed results load instantly. Hard cases show where BanglaBERT title only
            fails but full VTCF gets it right.
          </p>
          {examplesError && <p className="text-sm text-clickbait">{examplesError}</p>}
          {examples.length > 0 && (
            <ExampleCards
              examples={examples}
              onSelect={onExampleSelect}
              disabled={loading}
            />
          )}
        </div>

        <AnimatePresence mode="popLayout" initial={false}>
          {showResultsPanel && (
            <motion.section
              ref={resultsRef}
              layout
              key="analysis-results"
              id="analysis-results"
              initial={reducedMotion ? false : { opacity: 0, y: 56 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reducedMotion ? undefined : { opacity: 0, y: 32 }}
              transition={motionTransition(!!reducedMotion, SPRING_GENTLE)}
              className="scroll-mt-8"
            >
              <AnimatePresence mode="wait">
                {loading && (
                  <motion.div
                    key="pipeline-loading"
                    layout
                    initial={reducedMotion ? false : { opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={reducedMotion ? undefined : { opacity: 0, y: -16 }}
                    transition={motionTransition(!!reducedMotion, SPRING_GENTLE)}
                  >
                    <PipelineProgress
                      activeStageIndex={stageIndex}
                      completed={loadComplete}
                      estimate={timeEstimate}
                    />
                  </motion.div>
                )}

                {result && !loading && (
                  <motion.div
                    key={result.video_id}
                    layout
                    initial={reducedMotion ? false : { opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={reducedMotion ? undefined : { opacity: 0, y: 20 }}
                    transition={motionTransition(!!reducedMotion, SPRING_GENTLE)}
                  >
                    <ResultsView result={result} researchNote={researchNote} />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </ScrollReveal>
  );
});
