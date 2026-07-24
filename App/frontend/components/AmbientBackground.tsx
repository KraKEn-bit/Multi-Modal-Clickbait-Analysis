"use client";

/**
 * Detective / forensic ambient background — soft spotlight cones drift
 * across the viewport, plus subtle scanning-line grid motif.
 */
export function AmbientBackground() {
  return (
    <div className="ambient-bg" aria-hidden="true">
      <div className="ambient-spotlight ambient-spotlight--teal" />
      <div className="ambient-spotlight ambient-spotlight--coral" />
      <div className="ambient-spotlight ambient-spotlight--teal-secondary" />
      <div className="ambient-scan-grid" />
      <div className="ambient-scan-beam" />
      <div className="ambient-bg__grain" />
      <div className="ambient-bg__scrim" />
    </div>
  );
}
