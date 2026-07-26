"use client";

import { useRef, type ReactNode, type PointerEvent } from "react";
import { motion, useMotionValue, useReducedMotion, useSpring } from "framer-motion";

const MAX_PULL_PX = 5;

/**
 * Kokonut-style magnetic button — the label gently follows the pointer
 * while hovered. Pointer events only (never scroll-driven); transform
 * springs are compositor-friendly.
 */
export function MagneticButton({
  children,
  className,
  onClick,
}: {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
}) {
  const reducedMotion = useReducedMotion();
  const ref = useRef<HTMLButtonElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const springX = useSpring(x, { stiffness: 300, damping: 20 });
  const springY = useSpring(y, { stiffness: 300, damping: 20 });

  const handleMove = (event: PointerEvent<HTMLButtonElement>) => {
    if (reducedMotion || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const relX = (event.clientX - rect.left) / rect.width - 0.5;
    const relY = (event.clientY - rect.top) / rect.height - 0.5;
    x.set(relX * 2 * MAX_PULL_PX);
    y.set(relY * 2 * MAX_PULL_PX);
  };

  const handleLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.button
      ref={ref}
      type="button"
      onClick={onClick}
      onPointerMove={handleMove}
      onPointerLeave={handleLeave}
      whileTap={reducedMotion ? undefined : { scale: 0.97 }}
      style={{ x: springX, y: springY }}
      className={`transform-gpu will-change-transform ${className ?? ""}`}
    >
      {children}
    </motion.button>
  );
}
