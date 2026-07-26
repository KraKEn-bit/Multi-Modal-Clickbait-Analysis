"use client";

import { forwardRef, memo, type ReactNode, type RefObject } from "react";
import { motion, useReducedMotion, type Transition } from "framer-motion";
import { useLenisScroll } from "@/components/LenisProvider";
import { usePrefersReducedMotion } from "@/lib/motion";

/**
 * Shared viewport config — `once: true` means the IntersectionObserver
 * disconnects after the first reveal, so scrolling never re-renders
 * these components again.
 */
const VIEWPORT = { once: true, amount: 0.15, margin: "0px 0px -48px 0px" } as const;

const EASE_OUT: Transition = { duration: 0.5, ease: [0.16, 1, 0.3, 1] };

type Props = {
  children: ReactNode;
  className?: string;
  id?: string;
  delay?: number;
};

/**
 * Viewport-triggered fade + 15px translate-y reveal.
 * GPU-composited (`transform-gpu` + `will-change-transform`), fires once,
 * respects reduced motion.
 */
export const ScrollReveal = memo(
  forwardRef<HTMLElement, Props>(function ScrollReveal(
    { children, className, id, delay = 0 },
    ref,
  ) {
    const reducedMotion = useReducedMotion();

    return (
      <motion.section
        ref={ref}
        id={id}
        className={`transform-gpu will-change-transform ${className ?? ""}`}
        initial={reducedMotion ? false : { opacity: 0, y: 15 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={VIEWPORT}
        transition={reducedMotion ? { duration: 0 } : { ...EASE_OUT, delay }}
      >
        {children}
      </motion.section>
    );
  }),
);

/** Inner block reveal for staggered content inside a section. */
export const ScrollRevealItem = memo(function ScrollRevealItem({
  children,
  className,
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  const reducedMotion = useReducedMotion();

  return (
    <motion.div
      className={`transform-gpu will-change-transform ${className ?? ""}`}
      initial={reducedMotion ? false : { opacity: 0, y: 15 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={VIEWPORT}
      transition={reducedMotion ? { duration: 0 } : { ...EASE_OUT, delay }}
    >
      {children}
    </motion.div>
  );
});

export function scrollToSection(
  ref: RefObject<HTMLElement | null>,
  reducedMotion: boolean,
  lenisScrollTo?: (target: HTMLElement, options?: { offset?: number }) => void,
) {
  if (ref.current && lenisScrollTo && !reducedMotion) {
    lenisScrollTo(ref.current, { offset: -72 });
    return;
  }

  ref.current?.scrollIntoView({
    behavior: reducedMotion ? "auto" : "smooth",
    block: "start",
  });
}

export function useScrollToSection() {
  const reducedMotion = usePrefersReducedMotion();
  const { scrollTo } = useLenisScroll();

  return (ref: RefObject<HTMLElement | null>) => {
    scrollToSection(ref, reducedMotion, (target, options) => scrollTo(target, options));
  };
}
