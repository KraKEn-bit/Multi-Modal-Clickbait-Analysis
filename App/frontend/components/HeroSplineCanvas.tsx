"use client";

import Spline from "@splinetool/react-spline";

export const HERO_SPLINE_SCENE =
  "https://prod.spline.design/OdQm0sq5GueGF-sq/scene.splinecode";

export function HeroSplineCanvas() {
  return (
    <Spline
      scene={HERO_SPLINE_SCENE}
      className="h-full w-full origin-center scale-125 md:scale-150 [&_canvas]:!h-full [&_canvas]:!w-full"
    />
  );
}
