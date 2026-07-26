"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { X } from "lucide-react";

/**
 * Kokonut-style toast — bottom-right, spring in/out, mono copy.
 * Rendering is opacity/transform only; auto-dismiss is owned by
 * the caller so this stays a dumb presentational component.
 */
export function Toast({
  message,
  onDismiss,
}: {
  message: string | null;
  onDismiss: () => void;
}) {
  const reducedMotion = useReducedMotion();

  return (
    <div className="pointer-events-none fixed bottom-6 right-6 z-50">
      <AnimatePresence>
        {message && (
          <motion.div
            key="toast"
            initial={reducedMotion ? false : { opacity: 0, y: 16, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reducedMotion ? undefined : { opacity: 0, y: 8, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 380, damping: 30 }}
            className="pointer-events-auto flex max-w-sm transform-gpu items-start gap-3 rounded-xl border border-white/10 bg-surface px-4 py-3 shadow-[0_16px_40px_-12px_rgba(0,0,0,0.7)] will-change-transform"
            role="status"
          >
            <span
              className="mt-1 h-2 w-2 shrink-0 animate-pulse rounded-full bg-agent"
              aria-hidden
            />
            <p className="font-mono text-xs leading-relaxed text-foreground">
              {message}
            </p>
            <button
              type="button"
              onClick={onDismiss}
              className="shrink-0 rounded-md p-0.5 text-muted transition-colors duration-200 hover:text-foreground"
              aria-label="Dismiss notification"
            >
              <X className="h-3.5 w-3.5" aria-hidden />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
