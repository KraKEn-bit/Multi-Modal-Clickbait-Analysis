"use client";

import { useEffect, useState } from "react";
import type { Transition } from "framer-motion";

/** Primary spring — fluid Apple-like settle (stiffness 100, damping 15). */
export const SPRING_GENTLE: Transition = {
  type: "spring",
  stiffness: 100,
  damping: 15,
};

/** Snappier spring for hover overlays and micro-interactions. */
export const SPRING_SNAPPY: Transition = {
  type: "spring",
  stiffness: 260,
  damping: 22,
};

/** Bouncy pop for verdict badge and emphasis moments. */
export const SPRING_BOUNCY: Transition = {
  type: "spring",
  stiffness: 180,
  damping: 14,
};

/** Heavier drop-in for verdict banner. */
export const SPRING_DROP: Transition = {
  type: "spring",
  stiffness: 140,
  damping: 16,
  mass: 0.9,
};

/** Expanding panel spring for TDS score block. */
export const SPRING_EXPAND: Transition = {
  type: "spring",
  stiffness: 90,
  damping: 14,
  mass: 1.1,
};

/** Layout reflow when page height changes. */
export const SPRING_LAYOUT: Transition = {
  type: "spring",
  stiffness: 120,
  damping: 20,
  mass: 0.8,
};

export const INSTANT: Transition = { duration: 0 };

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  return reduced;
}

export function motionTransition(
  reduced: boolean,
  spring: Transition = SPRING_GENTLE,
): Transition {
  return reduced ? INSTANT : spring;
}
