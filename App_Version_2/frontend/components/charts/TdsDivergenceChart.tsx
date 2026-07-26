"use client";

import { memo, useEffect, useRef } from "react";
import { useInView, useReducedMotion } from "framer-motion";
import { animate, stagger, svg } from "animejs";

/**
 * Shadcn-style area chart: cumulative visual divergence across the video
 * timeline, ending at the real study means (clickbait 0.383 vs genuine
 * 0.644). Curves are drawn in with Anime.js SVG path drawing when the
 * chart first scrolls into view; area fills and end labels fade after.
 *
 * Curve shapes between checkpoints are illustrative; the endpoint means
 * are the measured values from the study.
 */

// Plot geometry: x 40→456, y 200 (0.0) → 16 (0.7)
const yFor = (v: number) => 200 - (v / 0.7) * 184;

const Y_GRID = [0.2, 0.4, 0.6];
const X_TICKS = [
  { x: 40, label: "Hook" },
  { x: 248, label: "Context" },
  { x: 456, label: "Delivery" },
];

const GENUINE_LINE = `M 40 200 C 144 198, 180 ${yFor(0.35)}, 248 ${yFor(0.35)} C 316 ${yFor(0.35)}, 392 ${yFor(0.644)}, 456 ${yFor(0.644)}`;
const CLICKBAIT_LINE = `M 40 200 C 144 199, 184 ${yFor(0.18)}, 248 ${yFor(0.18)} C 312 ${yFor(0.18)}, 396 ${yFor(0.383)}, 456 ${yFor(0.383)}`;

const GENUINE_AREA = `${GENUINE_LINE} L 456 200 L 40 200 Z`;
const CLICKBAIT_AREA = `${CLICKBAIT_LINE} L 456 200 L 40 200 Z`;

export const TdsDivergenceChart = memo(function TdsDivergenceChart() {
  const reducedMotion = useReducedMotion();
  const rootRef = useRef<HTMLDivElement>(null);
  const inView = useInView(rootRef, { amount: 0.4, once: true });
  const hasRun = useRef(false);

  useEffect(() => {
    if (!inView || reducedMotion || hasRun.current) return;
    const root = rootRef.current;
    if (!root) return;
    hasRun.current = true;

    const lines = root.querySelectorAll<SVGPathElement>("[data-chart-line]");
    const fills = root.querySelectorAll<SVGPathElement>("[data-chart-fill]");
    const marks = root.querySelectorAll<SVGGElement>("[data-chart-mark]");

    animate(svg.createDrawable(lines), {
      draw: ["0 0", "0 1"],
      opacity: { to: 1, duration: 1 },
      duration: 1500,
      delay: stagger(240),
      ease: "inOut(2)",
    });
    animate(fills, {
      opacity: [0, 1],
      duration: 900,
      delay: 1100,
      ease: "outQuad",
    });
    animate(marks, {
      opacity: [0, 1],
      scale: [0.5, 1],
      duration: 500,
      delay: stagger(160, { start: 1400 }),
      ease: "outBack",
    });
  }, [inView, reducedMotion]);

  const hidden = reducedMotion ? undefined : ({ opacity: 0 } as const);

  return (
    <div ref={rootRef}>
      <div className="flex flex-wrap items-center gap-4">
        <span className="flex items-center gap-2 text-xs font-semibold text-foreground/80">
          <span className="h-2.5 w-2.5 rounded-full bg-agent" aria-hidden />
          Genuine · mean TDS 0.644
        </span>
        <span className="flex items-center gap-2 font-mono text-xs text-foreground/80">
          <span className="h-2.5 w-2.5 rounded-full bg-clickbait" aria-hidden />
          Clickbait · mean TDS 0.383
        </span>
      </div>

      <svg
        viewBox="0 0 480 240"
        className="mt-4 w-full"
        role="img"
        aria-label="Area chart of visual divergence across the video timeline: genuine videos reach a mean temporal divergence score of 0.644 while clickbait reaches only 0.383"
      >
        <defs>
          <linearGradient id="tds-fill-genuine" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00ffa3" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#00ffa3" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="tds-fill-clickbait" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ff5c47" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#ff5c47" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Grid */}
        {Y_GRID.map((v) => (
          <g key={v}>
            <line
              x1="40"
              y1={yFor(v)}
              x2="456"
              y2={yFor(v)}
              stroke="rgba(15,17,21,0.06)"
              strokeDasharray="4 6"
            />
            <text
              x="32"
              y={yFor(v) + 3.5}
              textAnchor="end"
              className="fill-muted"
              fontSize="10"
            >
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        <line x1="40" y1="200" x2="456" y2="200" stroke="rgba(15,17,21,0.12)" />

        {/* X labels */}
        {X_TICKS.map(({ x, label }) => (
          <text
            key={label}
            x={x}
            y="222"
            textAnchor={x === 40 ? "start" : x === 456 ? "end" : "middle"}
            className="fill-muted"
            fontSize="11"
          >
            {label}
          </text>
        ))}

        {/* Area fills */}
        <path d={GENUINE_AREA} fill="url(#tds-fill-genuine)" data-chart-fill style={hidden} />
        <path d={CLICKBAIT_AREA} fill="url(#tds-fill-clickbait)" data-chart-fill style={hidden} />

        {/* Lines — drawn in by Anime.js */}
        <path
          d={GENUINE_LINE}
          fill="none"
          stroke="#00ffa3"
          strokeWidth="2.5"
          strokeLinecap="round"
          data-chart-line
          style={hidden}
        />
        <path
          d={CLICKBAIT_LINE}
          fill="none"
          stroke="#ff5c47"
          strokeWidth="2.5"
          strokeLinecap="round"
          data-chart-line
          style={hidden}
        />

        {/* Endpoint markers + value badges */}
        <g data-chart-mark style={hidden}>
          <circle cx="456" cy={yFor(0.644)} r="4" fill="#00ffa3" />
          <rect x="398" y={yFor(0.644) - 26} rx="6" width="52" height="18" fill="#00ffa3" />
          <text x="424" y={yFor(0.644) - 13} textAnchor="middle" fill="#04120c" fontSize="10" fontWeight="700">
            0.644
          </text>
        </g>
        <g data-chart-mark style={hidden}>
          <circle cx="456" cy={yFor(0.383)} r="4" fill="#ff5c47" />
          <rect x="398" y={yFor(0.383) - 26} rx="6" width="52" height="18" fill="#ff5c47" />
          <text x="424" y={yFor(0.383) - 13} textAnchor="middle" fill="#ffffff" fontSize="10" fontWeight="700">
            0.383
          </text>
        </g>
      </svg>
    </div>
  );
});
