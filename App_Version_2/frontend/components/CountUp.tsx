"use client";

import { useEffect, useRef } from "react";
import { useInView, useReducedMotion } from "framer-motion";
import { animate } from "animejs";

function format(value: number, decimals: number): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Anime.js-powered number counter. Renders the final value by default
 * (SSR / no-JS / reduced motion), then counts up from zero the first
 * time it scrolls into view.
 */
export function CountUp({
  to,
  decimals = 0,
  prefix = "",
  suffix = "",
  duration = 1500,
  className,
}: {
  to: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
  className?: string;
}) {
  const reducedMotion = useReducedMotion();
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { amount: 0.6, once: true });
  const hasRun = useRef(false);

  useEffect(() => {
    if (!inView || reducedMotion || hasRun.current) return;
    const el = ref.current;
    if (!el) return;
    hasRun.current = true;

    const counter = { value: 0 };
    const animation = animate(counter, {
      value: to,
      duration,
      ease: "outExpo",
      onUpdate: () => {
        el.textContent = `${prefix}${format(counter.value, decimals)}${suffix}`;
      },
    });
    return () => {
      animation.pause();
    };
  }, [inView, reducedMotion, to, decimals, prefix, suffix, duration]);

  return (
    <span ref={ref} className={className}>
      {prefix}
      {format(to, decimals)}
      {suffix}
    </span>
  );
}
