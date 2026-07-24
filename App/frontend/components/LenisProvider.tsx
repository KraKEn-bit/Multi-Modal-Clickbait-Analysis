"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import Lenis from "lenis";
import { usePrefersReducedMotion } from "@/lib/motion";

type LenisContextValue = {
  lenis: Lenis | null;
  scrollTo: (target: HTMLElement | string, options?: { offset?: number }) => void;
};

const LenisContext = createContext<LenisContextValue>({
  lenis: null,
  scrollTo: () => {},
});

export function useLenisScroll() {
  return useContext(LenisContext);
}

export function LenisProvider({ children }: { children: ReactNode }) {
  const reducedMotion = usePrefersReducedMotion();
  const [lenis, setLenis] = useState<Lenis | null>(null);
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    if (reducedMotion) {
      return;
    }

    const instance = new Lenis({
      duration: 1.15,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
      touchMultiplier: 1.35,
    });

    lenisRef.current = instance;
    setLenis(instance);

    let frame = 0;
    const raf = (time: number) => {
      instance.raf(time);
      frame = requestAnimationFrame(raf);
    };
    frame = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(frame);
      instance.destroy();
      lenisRef.current = null;
      setLenis(null);
    };
  }, [reducedMotion]);

  const scrollTo = useCallback(
    (target: HTMLElement | string, options?: { offset?: number }) => {
      const offset = options?.offset ?? 0;
      const instance = lenisRef.current;

      if (instance) {
        instance.scrollTo(target, { offset, duration: 1.1 });
        return;
      }

      if (typeof target === "string") {
        document.querySelector(target)?.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }

      target.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [],
  );

  return (
    <LenisContext.Provider value={{ lenis, scrollTo }}>
      {children}
    </LenisContext.Provider>
  );
}
