"use client";

/** One-time forensic scan line across the hero on page load. */
export function HeroScanBeam() {
  return (
    <div className="hero-scan" aria-hidden="true">
      <div className="hero-scan__line" />
      <div className="hero-scan__glow" />
    </div>
  );
}
