"use client";

import { memo, useEffect, useId, useRef } from "react";
import { useInView, useReducedMotion } from "framer-motion";
import { animate, stagger } from "animejs";
import type { AlignmentScores } from "@/lib/types";

/**
 * Per-video divergence chart — title↔frame alignment across the timeline.
 */

const W = 480;
const H = 268;
const PAD = { top: 28, right: 24, bottom: 52, left: 44 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;
const X = [
  PAD.left + PLOT_W * 0.12,
  PAD.left + PLOT_W * 0.5,
  PAD.left + PLOT_W * 0.88,
];
const Y_ZERO = PAD.top + PLOT_H * 0.55;
const Y_SCALE = 900;
const REF_VALUE = 0.05;

const LABELS = ["Hook", "Context", "Delivery"] as const;

function yFor(value: number): number {
  return Y_ZERO - value * Y_SCALE;
}

function valueLabelY(pointY: number): number {
  const gap = 16;
  if (pointY < PAD.top + 36) return pointY + gap + 6;
  if (pointY > Y_ZERO + 8) return pointY - gap;
  return pointY - gap;
}

export const CaseDivergenceChart = memo(function CaseDivergenceChart({
  alignment,
}: {
  alignment: AlignmentScores;
}) {
  const reducedMotion = useReducedMotion();
  const rootRef = useRef<HTMLDivElement>(null);
  const lineRef = useRef<SVGPathElement>(null);
  const inView = useInView(rootRef, { amount: 0.45, once: true });
  const gradientId = useId().replace(/:/g, "");

  const values = [alignment.hook, alignment.context, alignment.delivery];
  const points = values.map((v, i) => ({ x: X[i], y: yFor(v), value: v }));

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y.toFixed(1)}`)
    .join(" ");
  const refY = yFor(REF_VALUE);
  const areaPath = `M ${points[0].x} ${refY} ${points
    .map((p) => `L ${p.x} ${p.y.toFixed(1)}`)
    .join(" ")} L ${points[2].x} ${refY} Z`;

  useEffect(() => {
    if (!inView || reducedMotion) return;
    const root = rootRef.current;
    const line = lineRef.current;
    if (!root || !line) return;

    const length = line.getTotalLength();
    line.style.strokeDasharray = `${length}`;
    const drawState = { offset: length };
    const lineAnim = animate(drawState, {
      offset: 0,
      duration: 1100,
      ease: "inOutCubic",
      onUpdate: () => {
        line.style.strokeDashoffset = `${drawState.offset}`;
      },
    });

    const fills = root.querySelectorAll<SVGElement>("[data-chart-fade]");
    const fadeAnim = animate(fills, {
      opacity: [0, 1],
      duration: 700,
      delay: stagger(140, { start: 500 }),
      ease: "outCubic",
    });

    return () => {
      lineAnim.pause();
      fadeAnim.pause();
    };
  }, [inView, reducedMotion]);

  const hiddenStyle = reducedMotion ? undefined : { opacity: 0 };
  const plotRight = W - PAD.right;

  return (
    <div ref={rootRef} className="space-y-3">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Title-to-frame alignment across hook, context, and delivery"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ff5c47" stopOpacity="0.02" />
            <stop offset="100%" stopColor="#ff5c47" stopOpacity="0.2" />
          </linearGradient>
        </defs>

        {[0.25, 0.5, 0.75].map((t) => {
          const y = PAD.top + PLOT_H * t;
          return (
            <line
              key={t}
              x1={PAD.left}
              x2={plotRight}
              y1={y}
              y2={y}
              stroke="rgba(255,255,255,0.05)"
            />
          );
        })}

        <line
          x1={PAD.left}
          x2={plotRight}
          y1={refY}
          y2={refY}
          stroke="#00ffa3"
          strokeOpacity="0.45"
          strokeWidth="1"
          strokeDasharray="5 5"
        />

        <line
          x1={PAD.left}
          x2={plotRight}
          y1={Y_ZERO}
          y2={Y_ZERO}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="1"
          strokeDasharray="2 4"
        />

        <path d={areaPath} fill={`url(#${gradientId})`} data-chart-fade style={hiddenStyle} />

        <path
          ref={lineRef}
          d={linePath}
          fill="none"
          stroke="#00e5ff"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {points.map((p, i) => (
          <g key={LABELS[i]} data-chart-fade style={hiddenStyle}>
            <circle cx={p.x} cy={p.y} r="9" fill="rgba(0,229,255,0.12)" />
            <circle cx={p.x} cy={p.y} r="3.5" fill="#00e5ff" />
            <text
              x={p.x}
              y={valueLabelY(p.y)}
              textAnchor="middle"
              className="fill-foreground font-mono text-[10px] font-medium"
            >
              {p.value >= 0 ? "+" : ""}
              {p.value.toFixed(3)}
            </text>
            <text
              x={p.x}
              y={H - PAD.bottom + 20}
              textAnchor="middle"
              className="fill-muted text-[11px]"
            >
              {LABELS[i]}
            </text>
          </g>
        ))}
      </svg>

      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 px-1 text-xs text-muted">
        <span className="flex items-center gap-2">
          <span
            className="inline-block h-px w-5 border-t border-dashed border-agent/60"
            aria-hidden
          />
          Expected if title matched visuals
        </span>
        <span className="font-mono text-[10px] text-muted/80">0.00 = neutral</span>
      </div>
    </div>
  );
});
