"use client";

import { motion, useReducedMotion } from "framer-motion";
import { motionTransition, SPRING_SNAPPY, usePrefersReducedMotion } from "@/lib/motion";

export function VtcfLogoMark() {
  const reducedMotion = usePrefersReducedMotion();
  const framerReduced = useReducedMotion();

  return (
    <motion.div
      className="vtcf-logo group relative flex h-10 w-10 shrink-0 items-center justify-center"
      whileHover={framerReduced ? undefined : { scale: 1.06 }}
      whileTap={framerReduced ? undefined : { scale: 0.96 }}
      transition={motionTransition(!!framerReduced, SPRING_SNAPPY)}
      aria-hidden="true"
    >
      {/* Rotating scan ring */}
      {!reducedMotion && (
        <motion.div
          className="absolute inset-0 rounded-xl border border-dashed border-accent/25"
          animate={{ rotate: 360 }}
          transition={{ duration: 14, repeat: Infinity, ease: "linear" }}
        />
      )}

      {/* Glow pulse */}
      <motion.div
        className="absolute inset-0 rounded-xl bg-accent/20 blur-md"
        animate={
          framerReduced
            ? { opacity: 0.25 }
            : { opacity: [0.15, 0.45, 0.15], scale: [0.92, 1.05, 0.92] }
        }
        transition={
          framerReduced
            ? { duration: 0 }
            : { duration: 2.8, repeat: Infinity, ease: "easeInOut" }
        }
      />

      <svg
        className="relative z-10 h-10 w-10"
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <rect
          x="1"
          y="1"
          width="38"
          height="38"
          rx="9"
          className="fill-[#0E1420] stroke-accent/35 group-hover:stroke-accent/60 transition-colors"
          strokeWidth="1"
        />

        {/* Viewfinder corners */}
        {[
          "M 8 12 V 8 H 12",
          "M 28 8 H 32 V 12",
          "M 32 28 V 32 H 28",
          "M 12 32 H 8 V 28",
        ].map((d, i) => (
          <motion.path
            key={d}
            d={d}
            stroke="#00D2FF"
            strokeWidth="1.5"
            strokeLinecap="round"
            initial={framerReduced ? { pathLength: 1, opacity: 0.7 } : { pathLength: 0, opacity: 0.4 }}
            animate={{ pathLength: 1, opacity: 0.85 }}
            transition={
              framerReduced
                ? { duration: 0 }
                : { duration: 0.6, delay: i * 0.08, ease: "easeOut" }
            }
          />
        ))}

        {/* Crosshair */}
        <motion.line
          x1="20"
          y1="13"
          x2="20"
          y2="17"
          stroke="#00D2FF"
          strokeWidth="1"
          strokeOpacity="0.5"
          animate={framerReduced ? undefined : { opacity: [0.3, 0.8, 0.3] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <motion.line
          x1="20"
          y1="23"
          x2="20"
          y2="27"
          stroke="#00D2FF"
          strokeWidth="1"
          strokeOpacity="0.5"
          animate={framerReduced ? undefined : { opacity: [0.8, 0.3, 0.8] }}
          transition={{ duration: 2, repeat: Infinity }}
        />

        <text
          x="20"
          y="22.5"
          textAnchor="middle"
          className="fill-accent font-[family-name:var(--font-jetbrains)] text-[13px] font-bold"
        >
          V
        </text>
      </svg>

      {/* Hover scan line */}
      {!reducedMotion && (
        <motion.span
          className="pointer-events-none absolute left-1 right-1 z-20 h-px bg-gradient-to-r from-transparent via-accent to-transparent opacity-0 group-hover:opacity-80"
          initial={{ top: "20%" }}
          whileHover={{ top: ["15%", "85%", "15%"] }}
          transition={{ duration: 1.2, ease: "easeInOut" }}
        />
      )}
    </motion.div>
  );
}
