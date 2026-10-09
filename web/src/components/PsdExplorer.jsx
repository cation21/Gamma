import { useState } from "react";
import { cssVar, useCanvas } from "./useCanvas.js";

const BASE = { E: 1.5, O: 0, P: 0 };
const PRESETS = {
  base: { label: "Baseline", ...BASE },
  flat: { label: "Flatter background only", E: 1.0, O: 0, P: 0 },
  peak: { label: "Real gamma rhythm only", E: 1.5, O: 0, P: 0.35 },
};

const aperiodic = (f, s) => s.O - s.E * (Math.log10(f) - 1); // pivot at 10 Hz
const alpha = (f) => 0.5 * Math.exp(-((f - 10) ** 2) / 4.5);
const gamma = (f, s) => s.P * Math.exp(-((f - 85) ** 2) / 200);
const bandMean = (fn) => {
  let a = 0;
  let n = 0;
  for (let f = 60; f <= 110; f++) { a += fn(f); n++; }
  return a / n;
};

export default function PsdExplorer() {
  const [s, setS] = useState(BASE);

  const ref = useCanvas((g, W, H, d) => {
    const L = 48 * d, R = 14 * d, T = 14 * d, B = 38 * d, y0 = -2.6, y1 = 1.3;
    const X = (f) => L + (Math.log10(f) / Math.log10(120)) * (W - L - R);
    const Y = (v) => T + ((y1 - v) / (y1 - y0)) * (H - T - B);
    const mut = cssVar("--mut"), acc = cssVar("--acc");
    g.clearRect(0, 0, W, H);
    g.fillStyle = acc; g.globalAlpha = 0.14; g.fillRect(X(60), T, X(110) - X(60), H - T - B); g.globalAlpha = 1;
    g.fillStyle = mut; g.font = `${11 * d}px Inter, sans-serif`;
    g.fillText("gamma band 60 to 110 Hz", X(60) + 4 * d, T + 14 * d);
    g.strokeStyle = cssVar("--line"); g.lineWidth = d;
    g.beginPath(); g.moveTo(L, T); g.lineTo(L, H - B); g.lineTo(W - R, H - B); g.stroke();
    [1, 3, 10, 30, 100].forEach((f) => g.fillText(String(f), X(f) - 6 * d, H - B + 16 * d));
    g.fillText("Frequency (Hz, log scale)", L + (W - L - R) / 2 - 60 * d, H - 6 * d);
    g.save(); g.translate(14 * d, H / 2 + 40 * d); g.rotate(-Math.PI / 2); g.fillText("log10 power", 0, 0); g.restore();
    const line = (fn, color, width, dash) => {
      g.setLineDash(dash); g.strokeStyle = color; g.lineWidth = width; g.beginPath();
      for (let f = 1; f <= 120; f += 0.5) { const p = [X(f), Y(fn(f))]; f === 1 ? g.moveTo(...p) : g.lineTo(...p); }
      g.stroke(); g.setLineDash([]);
    };
    line((f) => aperiodic(f, s), mut, 1.5 * d, [6 * d, 5 * d]);
    line((f) => aperiodic(f, s) + alpha(f) + gamma(f, s), acc, 2.6 * d, []);
    g.fillStyle = mut; g.fillText("dashed: background (aperiodic)", L + 10 * d, H - B - 12 * d);
    g.fillStyle = acc; g.fillText("solid: what the sensor sees", L + 10 * d, H - B - 28 * d);
  }, [s]);

  const dA = bandMean((f) => aperiodic(f, s)) - bandMean((f) => aperiodic(f, BASE));
  const dP = bandMean((f) => gamma(f, s));
  const dB = dA + dP;
  const set = (k) => (e) => setS({ ...s, [k]: parseFloat(e.target.value) });

  return (
    <div className="grid">
      <canvas ref={ref} role="img" aria-label="Interactive power spectrum with background slope and optional gamma peak" />
      <div className="controls">
        <div className="btns">
          {Object.entries(PRESETS).map(([k, p]) => (
            <button key={k} onClick={() => setS({ E: p.E, O: p.O, P: p.P })}>{p.label}</button>
          ))}
        </div>
        <label>Exponent χ <span>{s.E.toFixed(2)}</span>
          <input type="range" min="0.5" max="2.5" step="0.01" value={s.E} onChange={set("E")} /></label>
        <label>Offset b <span>{s.O.toFixed(2)}</span>
          <input type="range" min="-0.5" max="0.5" step="0.01" value={s.O} onChange={set("O")} /></label>
        <label>Gamma peak height h <span>{s.P.toFixed(2)}</span>
          <input type="range" min="0" max="0.5" step="0.01" value={s.P} onChange={set("P")} /></label>
        <div className="readout">
          Band power change ΔB = <b>{dB.toFixed(2)}</b> log<sub>10</sub> units (<b>×{(10 ** dB).toFixed(1)}</b>)<br />
          = aperiodic part <b>{dA.toFixed(2)}</b> + periodic part <b>{dP.toFixed(2)}</b>
        </div>
      </div>
    </div>
  );
}
