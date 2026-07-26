"use client";

import { memo, useMemo } from "react";
import Particles, { ParticlesProvider } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";
import type { Engine, ISourceOptions } from "@tsparticles/engine";
import { usePrefersReducedMotion } from "@/lib/motion";

/** Module-level so the init callback stays stable across the app lifecycle. */
const initEngine = async (engine: Engine) => {
  await loadSlim(engine);
};

/**
 * Lightweight connecting-node constellation (tsparticles slim build).
 * Low particle count, links enabled, fps-capped, auto-paused when
 * off-screen or when the tab blurs. Memoized so parent re-renders
 * never touch the canvas. Skipped under reduced motion.
 */
export const ParticleNetwork = memo(function ParticleNetwork({
  id = "particle-network",
}: {
  id?: string;
}) {
  const reducedMotion = usePrefersReducedMotion();

  const options = useMemo<ISourceOptions>(
    () => ({
      fullScreen: { enable: false },
      fpsLimit: 60,
      detectRetina: true,
      pauseOnBlur: true,
      pauseOnOutsideViewport: true,
      background: { color: { value: "transparent" } },
      particles: {
        number: { value: 30, density: { enable: true } },
        color: { value: "#0d9488" },
        opacity: { value: 0.3 },
        size: { value: { min: 1, max: 2.5 } },
        links: {
          enable: true,
          distance: 150,
          color: "#0d9488",
          opacity: 0.14,
          width: 1,
        },
        move: {
          enable: true,
          speed: 0.5,
          outModes: { default: "bounce" },
        },
      },
    }),
    [],
  );

  if (reducedMotion) {
    return null;
  }

  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden="true">
      <ParticlesProvider init={initEngine}>
        <Particles id={id} options={options} className="h-full w-full" />
      </ParticlesProvider>
    </div>
  );
});
