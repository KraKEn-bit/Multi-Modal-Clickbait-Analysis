"use client";

import { forwardRef, type ReactNode, type RefObject } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { useLenisScroll } from "@/components/LenisProvider";
import { usePrefersReducedMotion } from "@/lib/motion";

type Props = {
  children: ReactNode;
  className?: string;
  id?: string;
  delay?: number;
};

/** Viewport-triggered fade + translate-y reveal. Respects reduced motion. */
export const ScrollReveal = forwardRef<HTMLElement, Props>(function ScrollReveal(
  { children, className, id, delay = 0 },
  ref,
) {
  const reducedMotion = useReducedMotion();

  return (
    <motion.section
      ref={ref}
      id={id}
      className={className}
      initial={reducedMotion ? false : { opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.12, margin: "-64px 0px -64px 0px" }}
      transition={
        reducedMotion
          ? { duration: 0 }
          : { duration: 0.55, ease: [0.16, 1, 0.3, 1], delay }
      }
    >
      {children}
    </motion.section>
  );
});

/** Inner block reveal for staggered content inside a section. */
export function ScrollRevealItem({
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
      className={className}
      initial={reducedMotion ? false : { opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={
        reducedMotion
          ? { duration: 0 }
          : { duration: 0.5, ease: [0.16, 1, 0.3, 1], delay }
      }
    >
      {children}
    </motion.div>
  );
}

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
