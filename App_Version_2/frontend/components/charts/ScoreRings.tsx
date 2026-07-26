"use client";

import { memo, useEffect, useRef } from "react";
import { useInView, useReducedMotion } from "framer-motion";
import { animate } from "animejs";
import { CountUp } from "@/components/CountUp";

/**
 * Ring gauges for the headline metrics: 99.63% F1 on the main test set,
 * and 100% rescue on the 9 hard failure cases. Arcs sweep in with
 * Anime.js (strokeDashoffset — paint-cheap, fires once on scroll) and
 * the numbers count up in sync.
 */

const R = 52;
const CIRCUMFERENCE = 2 * Math.PI * R;
/** Sub-100 scores need ≥ this empty arc or they read as a closed circle. */
const MIN_VISIBLE_GAP_PERCENT = 4;

/** Map metric → visible arc length (number in center stays exact). */
function visualFillPercent(percent: number): number {
  if (percent >= 100) return 100;
  const empty = 100 - percent;
  return empty >= MIN_VISIBLE_GAP_PERCENT ? percent : 100 - MIN_VISIBLE_GAP_PERCENT;
}

function Ring({
  percent,
  color,
  trackColor,
  children,
  label,
  sublabel,
}: {
  percent: number;
  color: string;
  trackColor: string;
  children: React.ReactNode;
  label: string;
  sublabel: string;
}) {
  const reducedMotion = useReducedMotion();
  const fill = visualFillPercent(percent);
  const filledLength = (fill / 100) * CIRCUMFERENCE;
  const gapLength = CIRCUMFERENCE - filledLength;
  const dashTarget = `${filledLength} ${gapLength}`;

  return (
    <div className="flex flex-col items-center text-center">
      <div className="relative h-36 w-36">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
          <circle cx="60" cy="60" r={R} fill="none" stroke={trackColor} strokeWidth="10" />
          <circle
            cx="60"
            cy="60"
            r={R}
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={reducedMotion ? dashTarget : `0 ${CIRCUMFERENCE}`}
            strokeDashoffset={0}
            data-ring-arc
            data-ring-target={dashTarget}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          {children}
        </div>
      </div>
      <p className="mt-3 text-sm font-bold text-foreground">{label}</p>
      <p className="mt-0.5 max-w-[180px] text-xs leading-relaxed text-muted">{sublabel}</p>
    </div>
  );
}

export const ScoreRings = memo(function ScoreRings() {
  const reducedMotion = useReducedMotion();
  const rootRef = useRef<HTMLDivElement>(null);
  const inView = useInView(rootRef, { amount: 0.4, once: true });
  const hasRun = useRef(false);

  useEffect(() => {
    if (!inView || reducedMotion || hasRun.current) return;
    const root = rootRef.current;
    if (!root) return;
    hasRun.current = true;

    const arcs = Array.from(root.querySelectorAll<SVGCircleElement>("[data-ring-arc]"));
    arcs.forEach((arc, i) => {
      const target = arc.dataset.ringTarget ?? `0 ${CIRCUMFERENCE}`;
      animate(arc, {
        strokeDasharray: [ `0 ${CIRCUMFERENCE}`, target ],
        duration: 1500,
        delay: 200 + i * 250,
        ease: "outQuart",
      });
    });
  }, [inView, reducedMotion]);

  return (
    <div
      ref={rootRef}
      className="grid grid-cols-1 items-start justify-items-center gap-8 sm:grid-cols-2"
    >
      <Ring
        percent={99.63}
        color="#00ffa3"
        trackColor="rgba(0,255,163,0.12)"
        label="F1 score"
        sublabel="Full VTCF model on the main test set"
      >
        <span className="font-mono text-2xl font-bold tracking-tight text-agent">
          <CountUp to={99.63} decimals={2} suffix="%" duration={1700} />
        </span>
      </Ring>

      <Ring
        percent={100}
        color="#00e5ff"
        trackColor="rgba(0,229,255,0.12)"
        label="Hard-case rescue"
        sublabel="All 9 text-blind failure cases recovered by visual evidence"
      >
        <span className="font-mono text-2xl font-bold tracking-tight text-accent">
          <CountUp to={100} suffix="%" duration={1700} />
        </span>
      </Ring>
    </div>
  );
});
