import { useEffect, useRef } from "react";

export const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Sizes the canvas to its CSS width at device resolution and calls draw(ctx, W, H, dpr).
// Redraws when deps change, on resize, and when the colour scheme flips.
export function useCanvas(draw, deps) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return undefined;
    const render = () => {
      const r = c.getBoundingClientRect();
      const d = window.devicePixelRatio || 1;
      c.width = Math.max(300, r.width * d);
      c.height = Math.max(260, r.width * 0.58) * d;
      c.style.height = `${c.height / d}px`;
      draw(c.getContext("2d"), c.width, c.height, d);
    };
    render();
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    window.addEventListener("resize", render);
    mq.addEventListener("change", render);
    return () => {
      window.removeEventListener("resize", render);
      mq.removeEventListener("change", render);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}
