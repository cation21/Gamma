import { useState } from "react";
import { cssVar, useCanvas } from "./useCanvas.js";

// Recovered mean differences and paired Cohen's d per scenario, from results/tables/sim_summary.csv.
const SCENARIOS = [
  { n: "No change", pl: "nothing", o: 0, e: 0, pk: 0, a: [-0.008, -0.05], p: [0, 0], x: [0.002, 0.03], b: [-0.004, -0.06], det: "0% to 0%" },
  { n: "True 80 Hz peak", pl: "gamma peak [80 Hz, 0.25, 8]", o: 0, e: 0, pk: 1, a: [0.103, 0.61], p: [0.093, 37.51], x: [-0.005, -0.06], b: [-0.002, -0.02], det: "0% to 100%" },
  { n: "Offset +0.15", pl: "offset +0.15", o: 0.15, e: 0, pk: 0, a: [0.154, 1.14], p: [0, -0.16], x: [-0.003, -0.05], b: [0.148, 2.34], det: "2.5% to 0%" },
  { n: "Exponent −0.25", pl: "exponent −0.25", o: 0, e: -0.25, pk: 0, a: [0.498, 3.04], p: [0, 0], x: [-0.259, -3.34], b: [0, -0.01], det: "0% to 0%" },
  { n: "Peak and exponent", pl: "peak and exponent −0.25", o: 0, e: -0.25, pk: 1, a: [0.581, 3.28], p: [0.093, 43.91], x: [-0.254, -3.15], b: [-0.001, -0.01], det: "0% to 100%" },
];

const gauss = (f, c, h, sd) => h * Math.exp(-((f - c) ** 2) / (2 * sd * sd));
const spectrum = (f, o, e, pk) =>
  1 + o - (1.5 + e) * Math.log10(f) + gauss(f, 10, 0.6, 2) + gauss(f, 20, 0.2, 3) + (pk ? gauss(f, 80, 0.25, 8) : 0);

function mulberry32(seed) {
  let s = seed;
  return () => {
    s |= 0; s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const sign = (v) => (v > 0 ? "+" : v < 0 ? "−" : "");
const f3 = (v) => sign(v) + Math.abs(v).toFixed(3);
const f2 = (v) => sign(v) + Math.abs(v).toFixed(2);

export default function SimExplorer() {
  const [cur, setCur] = useState(1);
  const s = SCENARIOS[cur];

  const ref = useCanvas((g, W, H, d) => {
    const L = 46 * d, R = 12 * d, T = 14 * d, B = 36 * d, y0 = -2.8, y1 = 0.9;
    const l3 = Math.log10(3), l2 = Math.log10(200);
    const X = (f) => L + ((Math.log10(f) - l3) / (l2 - l3)) * (W - L - R);
    const Y = (v) => T + ((y1 - v) / (y1 - y0)) * (H - T - B);
    const mut = cssVar("--mut"), acc = cssVar("--acc"), line = cssVar("--line");
    g.clearRect(0, 0, W, H);
    g.fillStyle = acc; g.globalAlpha = 0.14; g.fillRect(X(60), T, X(110) - X(60), H - T - B); g.globalAlpha = 1;
    g.strokeStyle = line; g.lineWidth = d; g.setLineDash([3 * d, 4 * d]);
    [30, 120].forEach((f) => { g.beginPath(); g.moveTo(X(f), T); g.lineTo(X(f), H - B); g.stroke(); });
    g.setLineDash([]);
    g.beginPath(); g.moveTo(L, T); g.lineTo(L, H - B); g.lineTo(W - R, H - B); g.stroke();
    g.fillStyle = mut; g.font = `${11 * d}px Inter, sans-serif`;
    [3, 10, 30, 100, 200].forEach((f) => g.fillText(String(f), X(f) - 6 * d, H - B + 16 * d));
    g.fillText("Frequency (Hz, log scale)", L + (W - L - R) / 2 - 60 * d, H - 6 * d);
    g.fillText("fit range 30 to 120 Hz", X(30) + 4 * d, T + 12 * d);
    g.save(); g.translate(14 * d, H / 2 + 40 * d); g.rotate(-Math.PI / 2); g.fillText("log10 power", 0, 0); g.restore();

    const rn = mulberry32(2026);
    const noise = () => Math.sqrt(-2 * Math.log(rn() + 1e-12)) * Math.cos(6.2832 * rn()) * 0.02;
    const trace = (fn, color, width, dash, alphaV = 1) => {
      g.globalAlpha = alphaV; g.setLineDash(dash); g.strokeStyle = color; g.lineWidth = width; g.beginPath();
      for (let f = 3, i = 0; f <= 200; f += 0.5, i++) { const p = [X(f), Y(fn(f))]; i ? g.lineTo(...p) : g.moveTo(...p); }
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    };
    trace((f) => spectrum(f, s.o, s.e, s.pk) + noise(), acc, d, [], 0.35);
    trace((f) => spectrum(f, 0, 0, 0), mut, 1.6 * d, [6 * d, 5 * d]);
    trace((f) => spectrum(f, s.o, s.e, s.pk), acc, 2.4 * d, []);
    g.fillStyle = mut; g.fillText("dashed: thinking", L + 8 * d, H - B - 30 * d);
    g.fillStyle = acc; g.fillText("solid: meditation (ground truth), thin: with noise 0.02", L + 8 * d, H - B - 14 * d);
  }, [cur]);

  return (
    <>
      <div className="btns" role="group" aria-label="Planted change">
        {SCENARIOS.map((sc, i) => (
          <button key={sc.n} aria-pressed={i === cur} onClick={() => setCur(i)}>{sc.n}</button>
        ))}
      </div>
      <div className="grid">
        <canvas ref={ref} role="img" aria-label="Simulated thinking and meditation spectra for the selected planted change" />
        <div className="readout">
          <b>{s.n}</b><br />
          Planted: {s.pl}<br />
          Naive gamma: <b>{f3(s.a[0])}</b> (d {f2(s.a[1])})<br />
          Periodic gamma: <b>{f3(s.p[0])}</b> (d {f2(s.p[1])})<br />
          Exponent: <b>{f3(s.x[0])}</b> (d {f2(s.x[1])})<br />
          Offset: <b>{f3(s.b[0])}</b> (d {f2(s.b[1])})<br />
          Gamma peak detected: <b>{s.det}</b>
        </div>
      </div>
    </>
  );
}
