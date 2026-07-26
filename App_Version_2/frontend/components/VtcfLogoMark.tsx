"use client";

import { memo } from "react";

/**
 * Static viewfinder logo for the dark agent header.
 * Single CSS-driven ring rotation (compositor-only).
 */
export const VtcfLogoMark = memo(function VtcfLogoMark() {
  return (
    <div
      className="relative flex h-10 w-10 shrink-0 items-center justify-center"
      aria-hidden="true"
    >
      <div className="logo-ring absolute inset-0 rounded-xl border border-dashed border-agent/25" />

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
          fill="#0e0e11"
          stroke="rgba(0, 255, 163, 0.35)"
          strokeWidth="1"
        />

        {[
          "M 8 12 V 8 H 12",
          "M 28 8 H 32 V 12",
          "M 32 28 V 32 H 28",
          "M 12 32 H 8 V 28",
        ].map((d) => (
          <path
            key={d}
            d={d}
            stroke="#00FFA3"
            strokeWidth="1.5"
            strokeLinecap="round"
            opacity="0.85"
          />
        ))}

        <line x1="20" y1="13" x2="20" y2="17" stroke="#00E5FF" strokeWidth="1" strokeOpacity="0.5" />
        <line x1="20" y1="23" x2="20" y2="27" stroke="#00E5FF" strokeWidth="1" strokeOpacity="0.5" />

        <text
          x="20"
          y="22.5"
          textAnchor="middle"
          fill="#00FFA3"
          fontSize="13"
          fontWeight="700"
          fontFamily="var(--font-jetbrains), monospace"
        >
          V
        </text>
      </svg>
    </div>
  );
});
