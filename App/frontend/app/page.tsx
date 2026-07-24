"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LayoutGroup, motion, useReducedMotion } from "framer-motion";
import { analyzeUrl, fetchAnalyzeEstimate, fetchExamples } from "@/lib/api";
import type { AnalysisResult, AnalyzeEstimate } from "@/lib/types";
import { AmbientBackground } from "@/components/AmbientBackground";
import { VtcfLogoMark } from "@/components/VtcfLogoMark";
import { HeroSection } from "@/components/landing/HeroSection";
import { HowItHelpsSection } from "@/components/landing/HowItHelpsSection";
import { ProblemSection } from "@/components/landing/ProblemSection";
import { ScienceSection } from "@/components/landing/ScienceSection";
import { ToolSection } from "@/components/landing/ToolSection";
import { useScrollToSection } from "@/components/ScrollReveal";
import { useLenisScroll } from "@/components/LenisProvider";
import { usePipelineStageTimer } from "@/components/PipelineProgress";
import { selectPreviewExample } from "@/lib/previewExample";
import { motionTransition, SPRING_LAYOUT } from "@/lib/motion";

export default function HomePage() {
  const [url, setUrl] = useState("");
  const [examples, setExamples] = useState<AnalysisResult[]>([]);
  const [researchNote, setResearchNote] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadComplete, setLoadComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [examplesError, setExamplesError] = useState<string | null>(null);
  const [timeEstimate, setTimeEstimate] = useState<AnalyzeEstimate | null>(null);

  const toolRef = useRef<HTMLElement>(null);
  const scienceRef = useRef<HTMLElement>(null);
  const resultsRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();
  const stageIndex = usePipelineStageTimer(loading, timeEstimate);
  const scrollToSectionFn = useScrollToSection();
  const { scrollTo: lenisScrollTo } = useLenisScroll();

  const previewExample = useMemo(
    () => selectPreviewExample(examples),
    [examples],
  );

  const scrollToResults = useCallback(() => {
    if (!resultsRef.current) return;
    if (!reducedMotion) {
      lenisScrollTo(resultsRef.current, { offset: -72 });
      return;
    }
    resultsRef.current.scrollIntoView({ behavior: "auto", block: "start" });
  }, [reducedMotion, lenisScrollTo]);

  useEffect(() => {
    fetchExamples()
      .then((data) => {
        setExamples(data.examples);
        setResearchNote(data.research_note);
      })
      .catch((err) => setExamplesError(err.message));
  }, []);

  useEffect(() => {
    if (loading || result) {
      const timer = window.setTimeout(
        () => scrollToResults(),
        reducedMotion ? 0 : 120,
      );
      return () => window.clearTimeout(timer);
    }
  }, [loading, result, scrollToResults, reducedMotion]);

  const handleAnalyze = useCallback(async () => {
    if (!url.trim()) return;
    setError(null);
    setResult(null);
    setLoading(true);
    setLoadComplete(false);
    setTimeEstimate(null);
    try {
      let estimate: AnalyzeEstimate | null = null;
      try {
        estimate = await fetchAnalyzeEstimate(url.trim());
        setTimeEstimate(estimate);
      } catch {
        setTimeEstimate({
          video_id: "",
          youtube_url: url.trim(),
          title: "",
          duration_seconds: null,
          duration_label: "unknown length",
          estimated_seconds_low: 45,
          estimated_seconds_high: 60,
          estimated_label: "45–60 sec",
        });
      }

      const data = await analyzeUrl(url.trim());
      setResult(data);
      setLoadComplete(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }, [url]);

  const handleExampleSelect = useCallback((example: AnalysisResult) => {
    setUrl(example.youtube_url);
    setError(null);
    setResult(example);
    setLoading(false);
    setLoadComplete(false);
    setTimeEstimate(null);
  }, []);

  return (
    <>
      <AmbientBackground />
      <LayoutGroup>
        <motion.div layout className="page-shell">
          <header className="sticky top-0 z-20 border-b border-border/60 bg-background/80 backdrop-blur">
            <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
              <div className="flex items-center gap-3">
                <VtcfLogoMark />
                <div>
                  <p className="text-sm font-semibold tracking-tight">
                    Visual-Temporal Contradiction Framework
                  </p>
                  <p className="text-[10px] uppercase tracking-widest text-muted">
                    Bangla clickbait detector
                  </p>
                </div>
              </div>
              <nav className="hidden items-center gap-6 text-xs text-muted sm:flex">
                <button
                  type="button"
                  onClick={() => scrollToSectionFn(toolRef)}
                  className="transition hover:text-accent"
                >
                  Analyze
                </button>
                <button
                  type="button"
                  onClick={() => scrollToSectionFn(scienceRef)}
                  className="transition hover:text-accent"
                >
                  Study
                </button>
              </nav>
            </div>
          </header>

          <motion.main
            layout
            transition={motionTransition(!!reducedMotion, SPRING_LAYOUT)}
            className="mx-auto max-w-6xl px-6"
          >
            <HeroSection
              onStartAnalyzing={() => scrollToSectionFn(toolRef)}
              onSeeStudy={() => scrollToSectionFn(scienceRef)}
            />

            <ToolSection
              ref={toolRef}
              resultsRef={resultsRef}
              url={url}
              onUrlChange={setUrl}
              onAnalyze={handleAnalyze}
              onExampleSelect={handleExampleSelect}
              loading={loading}
              loadComplete={loadComplete}
              error={error}
              examples={examples}
              examplesError={examplesError}
              researchNote={researchNote}
              result={result}
              timeEstimate={timeEstimate}
              stageIndex={stageIndex}
            />

            <ProblemSection />

            <HowItHelpsSection previewExample={previewExample} />

            <ScienceSection ref={scienceRef} />
          </motion.main>

          <motion.footer
            layout
            transition={motionTransition(!!reducedMotion, SPRING_LAYOUT)}
            className="mt-8 border-t border-border py-8 text-center text-xs text-muted"
          >
            VTCF study demo · BanglaBERT + ViT · Not affiliated with YouTube
          </motion.footer>
        </motion.div>
      </LayoutGroup>
    </>
  );
}
