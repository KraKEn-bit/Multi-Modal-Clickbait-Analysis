"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import { analyzeUrl, fetchAnalyzeEstimate, fetchExamples } from "@/lib/api";
import type { AnalysisResult, AnalyzeEstimate } from "@/lib/types";
import { HARD_CASES } from "@/lib/hardCases";
import { extractYoutubeVideoId } from "@/lib/youtube";
import { VtcfLogoMark } from "@/components/VtcfLogoMark";
import { Toast } from "@/components/Toast";
import { AgentConsole } from "@/components/landing/AgentConsole";
import { HeroSection } from "@/components/landing/HeroSection";
import { ValueSection } from "@/components/landing/ValueSection";
import { CaseStudyScroll } from "@/components/landing/CaseStudyScroll";
import { FindingsBento } from "@/components/landing/FindingsBento";
import { ScienceSection } from "@/components/landing/ScienceSection";
import { ToolSection } from "@/components/landing/ToolSection";
import { useScrollToSection } from "@/components/ScrollReveal";
import { useLenisScroll } from "@/components/LenisProvider";
import { usePipelineStageTimer } from "@/components/PipelineProgress";

const TOAST_DISMISS_MS = 8000;

/** Local hard cases shaped as AnalysisResult — always available without the API. */
const LOCAL_HARD_EXAMPLES: AnalysisResult[] = HARD_CASES.map((c) => ({
  video_id: c.video_id,
  youtube_url: `https://www.youtube.com/watch?v=${c.video_id}`,
  title: c.title,
  verdict: c.verdict,
  confidence: c.confidence,
  tds_score: c.tds_score,
  explanation: c.example_description,
  alignment_scores: c.alignment_scores,
  frame_urls: [...c.frame_urls],
  processing_time_seconds: 0,
  category: "hard",
  example_label: c.example_label,
  example_description: c.example_description,
  ground_truth: c.ground_truth,
  text_only: c.text_only,
  vtcf_rescued: true,
}));

export default function HomePage() {
  const [url, setUrl] = useState("");
  const [examples, setExamples] = useState<AnalysisResult[]>(LOCAL_HARD_EXAMPLES);
  const [researchNote, setResearchNote] = useState(
    "Clickbait videos show LESS visual change on average than genuine videos because clickbait often reuses static footage rather than bait-and-switching content.",
  );
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeEstimate, setTimeEstimate] = useState<AnalyzeEstimate | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const heroRef = useRef<HTMLDivElement>(null);
  const toolRef = useRef<HTMLElement>(null);
  const scienceRef = useRef<HTMLElement>(null);
  const resultsRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();
  const stageIndex = usePipelineStageTimer(loading, timeEstimate);
  const scrollToSectionFn = useScrollToSection();
  const { scrollTo: lenisScrollTo } = useLenisScroll();

  const hardCase = useMemo(
    () =>
      examples.find((e) => e.category === "hard" && e.text_only?.wrong) ??
      examples.find((e) => e.category === "hard") ??
      LOCAL_HARD_EXAMPLES[0],
    [examples],
  );

  const cachedById = useMemo(() => {
    const map = new Map<string, AnalysisResult>();
    for (const example of [...LOCAL_HARD_EXAMPLES, ...examples]) {
      map.set(example.video_id, example);
    }
    return map;
  }, [examples]);

  const scrollToResults = useCallback(() => {
    if (!resultsRef.current) return;
    if (!reducedMotion) {
      lenisScrollTo(resultsRef.current, { offset: -72 });
      return;
    }
    resultsRef.current.scrollIntoView({ behavior: "auto", block: "start" });
  }, [reducedMotion, lenisScrollTo]);

  // Enrich from the API when available; keep local hard cases if it fails.
  useEffect(() => {
    fetchExamples()
      .then((data) => {
        if (data.examples.length > 0) setExamples(data.examples);
        if (data.research_note) setResearchNote(data.research_note);
      })
      .catch(() => {
        /* silent — LOCAL_HARD_EXAMPLES already seeded */
      });
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), TOAST_DISMISS_MS);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const handleAnalyze = useCallback(async () => {
    if (!url.trim()) return;

    const pastedUrl = url.trim();
    setError(null);
    setToast(null);
    setResult(null);
    setTimeEstimate(null);

    const videoId = extractYoutubeVideoId(pastedUrl);
    if (!videoId) {
      setError("Paste a valid YouTube URL (watch, youtu.be, or shorts).");
      return;
    }

    // Instant path for pre-computed cases (works offline / on Vercel).
    const cached = cachedById.get(videoId);
    if (cached) {
      setLoading(false);
      setResult(cached);
      return;
    }

    // Live pipeline — same flow as before (stages + full report).
    setLoading(true);
    try {
      try {
        const estimate = await fetchAnalyzeEstimate(pastedUrl);
        setTimeEstimate(estimate);
      } catch {
        setTimeEstimate({
          video_id: videoId,
          youtube_url: pastedUrl,
          title: "",
          duration_seconds: null,
          duration_label: "",
          estimated_seconds_low: 120,
          estimated_seconds_high: 300,
          estimated_label: "a few minutes",
        });
      }

      const data = await analyzeUrl(pastedUrl);
      setResult(data);
    } catch (err) {
      // Keep the user's URL. Do not swap in a different hard case.
      const message =
        err instanceof Error
          ? err.message
          : "Analysis failed. Check the URL and try again.";
      setError(message);
      setToast(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [url, cachedById]);

  const handleExampleSelect = useCallback((example: AnalysisResult) => {
    setUrl(example.youtube_url);
    setError(null);
    setToast(null);
    setResult(example);
    setLoading(false);
    setTimeEstimate(null);
  }, []);

  const handleLoadDemo = useCallback(() => {
    if (hardCase) handleExampleSelect(hardCase);
  }, [hardCase, handleExampleSelect]);

  const scrollToConsole = useCallback(() => {
    if (!heroRef.current) return;
    if (!reducedMotion) {
      lenisScrollTo(heroRef.current, { offset: 0 });
      return;
    }
    heroRef.current.scrollIntoView({ behavior: "auto", block: "start" });
  }, [reducedMotion, lenisScrollTo]);

  return (
    <div className="page-shell">
      <div className="mesh-bg" aria-hidden />

      <header className="sticky top-0 z-20 border-b border-white/[0.07] bg-[#050505]">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <VtcfLogoMark />
            <div>
              <p className="font-mono text-sm font-bold tracking-tight text-white">
                VTCF
              </p>
              <p className="text-[11px] text-zinc-500">Bangla clickbait agent</p>
            </div>
          </div>
          <nav className="hidden items-center gap-2 sm:flex">
            <button
              type="button"
              onClick={() => scrollToSectionFn(scienceRef)}
              className="rounded-lg px-4 py-2 text-sm font-medium text-zinc-400 transition-colors duration-200 hover:text-accent"
            >
              The study
            </button>
            <button
              type="button"
              onClick={scrollToConsole}
              className="rounded-lg bg-agent px-4 py-2 text-sm font-semibold text-[#04120c] transition-colors duration-200 hover:bg-[#4dffbe]"
            >
              Analyze a video
            </button>
          </nav>
        </div>
      </header>

      <div ref={heroRef}>
        <HeroSection
          onSeeStudy={() => scrollToSectionFn(scienceRef)}
          console={
            <AgentConsole
              url={url}
              onUrlChange={setUrl}
              onAnalyze={handleAnalyze}
              onLoadDemo={handleLoadDemo}
              onViewReport={scrollToResults}
              loading={loading}
              fallback={false}
              error={error}
              stageIndex={stageIndex}
              estimate={timeEstimate}
              result={result}
              hasDemo={!!hardCase}
            />
          }
        />
      </div>

      <main>
        <CaseStudyScroll />

        <ToolSection
          ref={toolRef}
          resultsRef={resultsRef}
          onExampleSelect={handleExampleSelect}
          loading={loading}
          examples={examples}
          examplesError={null}
          researchNote={researchNote}
          result={result}
        />

        <ValueSection />

        <FindingsBento />

        <ScienceSection ref={scienceRef} />
      </main>

      <footer className="border-t border-white/[0.07] py-10 text-center font-mono text-xs text-muted">
        VTCF study demo · BanglaBERT + ViT · Not affiliated with YouTube
      </footer>

      <Toast message={toast} onDismiss={() => setToast(null)} />
    </div>
  );
}
