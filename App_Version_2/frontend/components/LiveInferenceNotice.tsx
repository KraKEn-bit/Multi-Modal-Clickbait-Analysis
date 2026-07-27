"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Cpu, X } from "lucide-react";

export const LIVE_UNAVAILABLE_MESSAGE =
  "Live inference runs on a dedicated GPU cluster. Please select one of our pre-computed hard cases below to explore the agent's full visual-temporal breakdown.";

type Props = {
  open: boolean;
  onDismiss: () => void;
  onBrowseCases: () => void;
};

export function LiveInferenceNotice({ open, onDismiss, onBrowseCases }: Props) {
  const reducedMotion = useReducedMotion();

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="live-inference-notice"
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          initial={reducedMotion ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={reducedMotion ? undefined : { opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          <button
            type="button"
            className="absolute inset-0 bg-black/70"
            aria-label="Dismiss notice"
            onClick={onDismiss}
          />

          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="live-inference-title"
            initial={reducedMotion ? false : { opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reducedMotion ? undefined : { opacity: 0, y: 10, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 380, damping: 32 }}
            className="relative z-10 w-full max-w-md rounded-2xl border border-white/10 bg-[#0d0d10] p-6 shadow-[0_24px_80px_-20px_rgba(0,0,0,0.85)]"
          >
            <button
              type="button"
              onClick={onDismiss}
              className="absolute right-4 top-4 rounded-md p-1 text-muted transition-colors duration-200 hover:text-foreground"
              aria-label="Close"
            >
              <X className="h-4 w-4" aria-hidden />
            </button>

            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-accent-soft">
              <Cpu className="h-5 w-5 text-accent" aria-hidden />
            </div>

            <h2
              id="live-inference-title"
              className="pr-8 text-base font-semibold tracking-tight text-foreground"
            >
              Live analysis unavailable here
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              {LIVE_UNAVAILABLE_MESSAGE}
            </p>

            <div className="mt-6 flex flex-col gap-2 sm:flex-row sm:justify-end">
              <button
                type="button"
                onClick={onDismiss}
                className="rounded-xl px-4 py-2.5 text-sm font-medium text-zinc-400 transition-colors duration-200 hover:text-foreground"
              >
                Dismiss
              </button>
              <button
                type="button"
                onClick={onBrowseCases}
                className="rounded-xl bg-agent px-4 py-2.5 text-sm font-semibold text-[#04120c] transition-colors duration-200 hover:bg-[#4dffbe]"
              >
                Browse hard cases
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
