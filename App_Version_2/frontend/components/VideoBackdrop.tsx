"use client";

import { memo } from "react";
import { useReducedMotion } from "framer-motion";

/**
 * GPU-friendly looping video backdrop for complex 3D visuals.
 * Accepts a `.webm` asset; when no asset is provided (or the user
 * prefers reduced motion) it falls back to a static pure-CSS mesh
 * gradient so there is never a heavy canvas in the background.
 */
export const VideoBackdrop = memo(function VideoBackdrop({
  src,
  poster,
  className = "",
}: {
  src?: string;
  poster?: string;
  className?: string;
}) {
  const reducedMotion = useReducedMotion();

  if (!src || reducedMotion) {
    return (
      <div
        className={`pointer-events-none absolute inset-0 ${className}`}
        aria-hidden="true"
        style={{
          background:
            "radial-gradient(ellipse 55% 45% at 25% 10%, rgba(0,255,163,0.08), transparent 70%), radial-gradient(ellipse 55% 50% at 80% 25%, rgba(0,229,255,0.07), transparent 70%)",
        }}
      />
    );
  }

  return (
    <video
      className={`pointer-events-none absolute inset-0 h-full w-full transform-gpu object-cover ${className}`}
      src={src}
      poster={poster}
      autoPlay
      loop
      muted
      playsInline
      preload="metadata"
      aria-hidden="true"
    />
  );
});
