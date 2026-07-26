"use client";

import { memo } from "react";

/**
 * Abstract geometric constellation for the charcoal hero.
 * Static SVG painted once; the only motion is a compositor-friendly
 * opacity twinkle on a handful of nodes. Zero JS per frame.
 */

const NODES: Array<{ x: number; y: number; r: number; twinkle?: 1 | 2 | 3 }> = [
  { x: 6, y: 22, r: 1.4, twinkle: 1 },
  { x: 14, y: 60, r: 1 },
  { x: 20, y: 34, r: 1.8, twinkle: 2 },
  { x: 30, y: 14, r: 1.1 },
  { x: 36, y: 48, r: 1.4, twinkle: 3 },
  { x: 47, y: 26, r: 1 },
  { x: 55, y: 56, r: 1.6, twinkle: 1 },
  { x: 63, y: 18, r: 1.2 },
  { x: 70, y: 42, r: 1.9, twinkle: 2 },
  { x: 79, y: 62, r: 1 },
  { x: 85, y: 28, r: 1.4, twinkle: 3 },
  { x: 93, y: 50, r: 1.1 },
];

const LINKS: Array<[number, number]> = [
  [0, 2],
  [1, 2],
  [2, 3],
  [2, 4],
  [3, 5],
  [4, 5],
  [4, 6],
  [5, 7],
  [6, 8],
  [7, 8],
  [8, 9],
  [8, 10],
  [10, 11],
  [9, 11],
];

const TWINKLE_CLASS = {
  1: "constellation-node",
  2: "constellation-node constellation-node--slow",
  3: "constellation-node constellation-node--slower",
} as const;

export const HeroConstellation = memo(function HeroConstellation() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox="0 0 100 80"
        preserveAspectRatio="xMidYMid slice"
      >
        {LINKS.map(([a, b]) => (
          <line
            key={`${a}-${b}`}
            x1={NODES[a].x}
            y1={NODES[a].y}
            x2={NODES[b].x}
            y2={NODES[b].y}
            stroke="rgba(45, 212, 191, 0.12)"
            strokeWidth="0.12"
          />
        ))}
        {NODES.map((node, i) => (
          <circle
            key={i}
            cx={node.x}
            cy={node.y}
            r={node.r * 0.28}
            fill="rgba(45, 212, 191, 0.55)"
            className={node.twinkle ? TWINKLE_CLASS[node.twinkle] : undefined}
            opacity={node.twinkle ? undefined : 0.3}
          />
        ))}
      </svg>

      {/* Soft vignette so nodes fade toward content */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 70% 60% at 50% 45%, rgba(15,17,21,0.55) 0%, #0f1115 88%)",
        }}
      />
    </div>
  );
});
