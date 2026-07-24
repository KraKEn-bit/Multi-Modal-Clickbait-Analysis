"use client";

import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Link2, Radar } from "lucide-react";
import { motionTransition, SPRING_GENTLE, SPRING_SNAPPY } from "@/lib/motion";

type Props = {
  url: string;
  onUrlChange: (url: string) => void;
  onAnalyze: () => void;
  loading: boolean;
  error: string | null;
};

export function AnalyzeLivePanel({
  url,
  onUrlChange,
  onAnalyze,
  loading,
  error,
}: Props) {
  const reducedMotion = useReducedMotion();
  const [focused, setFocused] = useState(false);
  const canSubmit = !loading && url.trim().length > 0;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (canSubmit) onAnalyze();
  };

  return (
    <motion.div
      className="analyze-panel relative overflow-hidden rounded-2xl"
      initial={reducedMotion ? false : { opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={motionTransition(!!reducedMotion, SPRING_GENTLE)}
    >
      <div className="analyze-panel__ambient" aria-hidden="true" />
      {!reducedMotion && <div className="analyze-panel__scan" aria-hidden="true" />}

      <form onSubmit={handleSubmit} className="relative z-10 p-6 sm:p-8">
        <div className="mb-5 flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-accent/30 bg-accent/10">
            <Radar className="h-5 w-5 text-accent" aria-hidden />
          </span>
          <div>
            <label htmlFor="youtube-url" className="text-sm font-semibold text-foreground">
              YouTube URL
            </label>
            <p className="text-xs text-muted">Paste a link to run live VTCF analysis</p>
          </div>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
          <div
            className={`analyze-input-wrap relative flex-1 ${focused ? "analyze-input-wrap--active" : ""}`}
          >
            <Link2
              className={`pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 transition-colors ${
                focused ? "text-accent" : "text-muted"
              }`}
              aria-hidden
            />
            <input
              id="youtube-url"
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={url}
              onChange={(e) => onUrlChange(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              disabled={loading}
              className="analyze-input w-full rounded-xl py-3.5 pl-11 pr-4 text-sm text-foreground outline-none disabled:opacity-50"
            />
          </div>

          <motion.button
            type="submit"
            disabled={!canSubmit}
            whileHover={reducedMotion || !canSubmit ? undefined : { scale: 1.03, y: -2 }}
            whileTap={reducedMotion || !canSubmit ? undefined : { scale: 0.97 }}
            transition={motionTransition(!!reducedMotion, SPRING_SNAPPY)}
            className={`analyze-submit shrink-0 rounded-xl px-7 py-3.5 text-sm font-bold tracking-wide transition disabled:cursor-not-allowed ${
              canSubmit ? "analyze-submit--ready" : "analyze-submit--idle"
            } ${loading ? "analyze-submit--loading" : ""}`}
          >
            {loading ? "Analyzing…" : "Analyze live"}
          </motion.button>
        </div>

        {error && (
          <motion.p
            initial={reducedMotion ? false : { opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 text-sm text-clickbait"
          >
            {error}
          </motion.p>
        )}

        <p className="mt-4 text-xs leading-relaxed text-foreground/70">
          Works best with videos under ~5 minutes for a quick live demo. Longer videos are
          supported but take longer to download and analyze.
        </p>
      </form>
    </motion.div>
  );
}
