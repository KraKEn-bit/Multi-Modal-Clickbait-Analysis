"use client";

import { useMemo } from "react";
import { motion, useReducedMotion } from "framer-motion";

/** Left cluster — title / text modality */
const TEXT_NODES = [
  { id: "t0", x: 11, y: 16, pulse: null },
  { id: "t1", x: 17, y: 26, pulse: "teal" as const },
  { id: "t2", x: 12, y: 36, pulse: null },
  { id: "t3", x: 19, y: 46, pulse: null },
  { id: "t4", x: 13, y: 56, pulse: "coral" as const },
  { id: "t5", x: 18, y: 66, pulse: null },
  { id: "t6", x: 14, y: 76, pulse: null },
  { id: "t7", x: 20, y: 86, pulse: "teal" as const },
];

/** Right cluster — visual / frame modality */
const VISUAL_NODES = [
  { id: "v0", x: 82, y: 14, pulse: null },
  { id: "v1", x: 76, y: 24, pulse: null },
  { id: "v2", x: 84, y: 34, pulse: "coral" as const },
  { id: "v3", x: 78, y: 44, pulse: null },
  { id: "v4", x: 83, y: 54, pulse: "teal" as const },
  { id: "v5", x: 77, y: 64, pulse: null },
  { id: "v6", x: 85, y: 74, pulse: null },
  { id: "v7", x: 79, y: 84, pulse: null },
];

/** Sparse cross-attention edges (text index → visual index) */
const ATTENTION_EDGES: Array<{
  from: number;
  to: number;
  pulse: "teal" | "coral" | null;
}> = [
  { from: 0, to: 2, pulse: null },
  { from: 0, to: 5, pulse: null },
  { from: 1, to: 1, pulse: "teal" },
  { from: 1, to: 4, pulse: null },
  { from: 2, to: 0, pulse: null },
  { from: 2, to: 6, pulse: "coral" },
  { from: 3, to: 3, pulse: null },
  { from: 3, to: 7, pulse: null },
  { from: 4, to: 2, pulse: "teal" },
  { from: 4, to: 5, pulse: null },
  { from: 5, to: 1, pulse: null },
  { from: 5, to: 4, pulse: "coral" },
  { from: 6, to: 6, pulse: null },
  { from: 7, to: 0, pulse: null },
  { from: 7, to: 3, pulse: "teal" },
];

const DRIFT_SEEDS = TEXT_NODES.concat(VISUAL_NODES).map((_, i) => ({
  dx: ((i * 7) % 5) - 2,
  dy: ((i * 11) % 5) - 2,
  duration: 16 + (i % 6) * 3,
  delay: (i % 5) * 0.8,
}));

function edgePath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): string {
  const mx = (x1 + x2) / 2;
  return `M ${x1} ${y1} Q ${mx} ${(y1 + y2) / 2} ${x2} ${y2}`;
}

export function HeroAttentionMatrix() {
  const reducedMotion = useReducedMotion();

  const edges = useMemo(
    () =>
      ATTENTION_EDGES.map((edge, i) => {
        const from = TEXT_NODES[edge.from];
        const to = VISUAL_NODES[edge.to];
        return {
          ...edge,
          key: `e${i}`,
          d: edgePath(from.x, from.y, to.x, to.y),
        };
      }),
    [],
  );

  const allNodes = useMemo(
    () => [
      ...TEXT_NODES.map((n, i) => ({ ...n, side: "text" as const, index: i })),
      ...VISUAL_NODES.map((n, i) => ({ ...n, side: "visual" as const, index: i + TEXT_NODES.length })),
    ],
    [],
  );

  return (
    <div
      className={`hero-matrix ${reducedMotion ? "hero-matrix--static" : ""}`}
      aria-hidden="true"
    >
      <svg
        className="hero-matrix__svg"
        viewBox="0 0 100 100"
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Intra-cluster faint links */}
        <g className="hero-matrix__intra" opacity={0.35}>
          {TEXT_NODES.slice(0, -1).map((node, i) => (
            <line
              key={`ti${i}`}
              x1={node.x}
              y1={node.y}
              x2={TEXT_NODES[i + 1].x}
              y2={TEXT_NODES[i + 1].y}
              stroke="#5B6D66"
              strokeWidth="0.12"
            />
          ))}
          {VISUAL_NODES.slice(0, -1).map((node, i) => (
            <line
              key={`vi${i}`}
              x1={node.x}
              y1={node.y}
              x2={VISUAL_NODES[i + 1].x}
              y2={VISUAL_NODES[i + 1].y}
              stroke="#5B6D66"
              strokeWidth="0.12"
            />
          ))}
        </g>

        {/* Cross-attention edges */}
        <g className="hero-matrix__cross">
          {edges.map((edge, i) =>
            reducedMotion ? (
              <path
                key={edge.key}
                d={edge.d}
                fill="none"
                stroke={
                  edge.pulse === "teal"
                    ? "rgba(0, 210, 255, 0.35)"
                    : edge.pulse === "coral"
                      ? "rgba(240, 101, 74, 0.3)"
                      : "rgba(91, 109, 102, 0.4)"
                }
                strokeWidth="0.1"
              />
            ) : (
              <motion.path
                key={edge.key}
                d={edge.d}
                fill="none"
                stroke={
                  edge.pulse === "teal"
                    ? "rgba(0, 210, 255, 0.5)"
                    : edge.pulse === "coral"
                      ? "rgba(240, 101, 74, 0.45)"
                      : "rgba(91, 109, 102, 0.35)"
                }
                strokeWidth="0.1"
                initial={{ opacity: 0.25 }}
                animate={{
                  opacity: edge.pulse
                    ? [0.2, 0.55, 0.2]
                    : [0.18, 0.32, 0.18],
                }}
                transition={{
                  duration: 5 + (i % 4) * 1.5,
                  repeat: Infinity,
                  ease: "easeInOut",
                  delay: i * 0.25,
                }}
              />
            ),
          )}
        </g>

        {/* Nodes */}
        {allNodes.map((node) => {
          const seed = DRIFT_SEEDS[node.index];
          const fill =
            node.pulse === "teal"
              ? "rgba(0, 210, 255, 0.55)"
              : node.pulse === "coral"
                ? "rgba(240, 101, 74, 0.5)"
                : "#0E1420";
          const stroke =
            node.pulse === "teal"
              ? "rgba(0, 210, 255, 0.7)"
              : node.pulse === "coral"
                ? "rgba(240, 101, 74, 0.65)"
                : "rgba(91, 109, 102, 0.55)";

          if (reducedMotion) {
            return (
              <circle
                key={node.id}
                cx={node.x}
                cy={node.y}
                r={node.pulse ? 0.55 : 0.42}
                fill={fill}
                stroke={stroke}
                strokeWidth="0.12"
              />
            );
          }

          return (
            <motion.g
              key={node.id}
              initial={{ x: 0, y: 0 }}
              animate={{
                x: [0, seed.dx * 0.6, seed.dx * -0.3, 0],
                y: [0, seed.dy * 0.5, seed.dy * -0.4, 0],
              }}
              transition={{
                duration: seed.duration,
                repeat: Infinity,
                ease: "easeInOut",
                delay: seed.delay,
              }}
            >
              <motion.circle
                cx={node.x}
                cy={node.y}
                r={node.pulse ? 0.55 : 0.42}
                fill={fill}
                stroke={stroke}
                strokeWidth="0.12"
                animate={
                  node.pulse
                    ? { opacity: [0.45, 0.85, 0.45], r: [0.42, 0.58, 0.42] }
                    : { opacity: [0.35, 0.55, 0.35] }
                }
                transition={{
                  duration: 4 + (node.index % 3) * 1.2,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
              />
            </motion.g>
          );
        })}
      </svg>

      <div className="hero-matrix__vignette" />
    </div>
  );
}
