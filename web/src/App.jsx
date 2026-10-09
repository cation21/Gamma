import { useEffect, useRef } from "react";
import PsdExplorer from "./components/PsdExplorer.jsx";
import SimExplorer from "./components/SimExplorer.jsx";

const REPO = "https://github.com/cation21/Gamma";

const Cite = ({ n }) => <a className="cite" href={`#r${n}`}>[{n}]</a>;

const Math_ = ({ children, note }) => (
  <div className="math">
    {children}
    {note && <small>{note}</small>}
  </div>
);

const Section = ({ kicker, title, children }) => (
  <section>
    <p className="kicker">{kicker}</p>
    <h2>{title}</h2>
    {children}
  </section>
);

const Step = ({ title, children }) => (
  <details>
    <summary>{title}</summary>
    <div>{children}</div>
  </details>
);

function Progress() {
  const ref = useRef(null);
  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      if (ref.current) ref.current.style.width = `${max > 0 ? (h.scrollTop / max) * 100 : 0}%`;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return <div className="progress" ref={ref} />;
}

const Table = ({ head, rows }) => (
  <div className="table-wrap">
    <table>
      <thead><tr>{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
      <tbody>
        {rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}
      </tbody>
    </table>
  </div>
);

const REFERENCES = [
  "Lutz A, Greischar LL, Rawlings NB, Ricard M, Davidson RJ (2004). Long-term meditators self-induce high-amplitude gamma synchrony during mental practice. PNAS.",
  "Braboszcz C, Cahn BR, Levy J, Fernandez M, Delorme A (2017). Increased gamma brainwave amplitude compared to control in three different meditation traditions. PLoS ONE. Data: OpenNeuro ds003969 (CC0).",
  "Whitham EM et al. (2007). Scalp electrical recording during paralysis: quantitative evidence that EEG frequencies above 20 Hz are contaminated by EMG. Clin Neurophysiol. See also Muthukumaraswamy SD (2013). High-frequency brain activity and muscle artifacts in MEG/EEG: a review and recommendations. Front Hum Neurosci.",
  "Donoghue T et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. Nat Neurosci.",
  "Gao R, Peterson EJ, Voytek B (2017). Inferring synaptic excitation/inhibition balance from field potentials. NeuroImage.",
  "Voytek B et al. (2015). Age-related changes in 1/f neural electrophysiological noise. J Neurosci.",
  "Colombo MA et al. (2019). The spectral exponent of the resting EEG indexes the presence of consciousness during unresponsiveness induced by propofol, xenon, and ketamine. NeuroImage.",
  "Lendner JD et al. (2020). An electrophysiological marker of arousal level in humans. eLife.",
  "Wen H, Liu Z (2016). Separating fractal and oscillatory components in the power spectrum of neurophysiological signal. Brain Topography.",
  "Pion-Tonachini L, Kreutz-Delgado K, Makeig S (2019). ICLabel: an automated electroencephalographic independent component classifier. NeuroImage.",
  "Gerster M et al. (2022). Separating neural oscillations from aperiodic 1/f activity: challenges and recommendations. Neuroinformatics.",
  "Lakens D (2017). Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. Soc Psychol Personal Sci.",
  "Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B.",
  "Welch PD (1967). The use of fast Fourier transform for the estimation of power spectra. IEEE Trans Audio Electroacoust.",
  <>This project: <a href={REPO}>github.com/cation21/Gamma</a>. Files: configs/default.yaml, results/REPORT.md, results/tables/*.csv (adjusted p-values in stats_mixed_2x2.csv, columns p_fdr_bh and p_holm).</>,
];

export default function App() {
  return (
    <>
      <Progress />

      <header className="hero">
        <div className="hero-inner">
          <div className="ribbon">September 2026 | Ananya Pandey</div>
          <div className="hero-row">
            <a className="logo logo-left" href="https://sites.google.com/iitmandi.ac.in/cuttinggardens-mandi-2026/home" target="_blank" rel="noopener noreferrer">
              <img src="/logos/cuttinggardens-2026.png" alt="CuttingGardens 2026" />
            </a>
            <div className="title">
              <h1>Periodic vs <em>aperiodic</em> gamma in meditation EEG</h1>
            </div>
            <a className="logo logo-right" href="https://www.iitmandi.ac.in/" target="_blank" rel="noopener noreferrer">
              <img src="/logos/iit-mandi.png" alt="Indian Institute of Technology Mandi" />
            </a>
          </div>
          <p className="sub">A specparam re-analysis of OpenNeuro ds003969, as a guided walkthrough: the claim, the maths, every pipeline step, the ground-truth test, and the results. Superscript numbers link to the reference list at the bottom. Open the "maths" boxes for formulas and the exact settings used.</p>
          <p className="presented">Poster presented at <b>CuttingGardens 2026, IIT Mandi</b>.</p>
        </div>
      </header>

      <Section kicker="1. Background" title={<>Gamma power in <em>meditation EEG</em></>}>
        <p>Gamma here means fast EEG activity, roughly 30 to 110 Hz. Experienced Tibetan practitioners showed high amplitude gamma during compassion meditation<Cite n={1} />. In the dataset used here, three meditation traditions showed higher gamma than controls<Cite n={2} />. "More gamma" means more <b>power</b> in that band. It does not say what produced the extra power.</p>
        <p>That matters because EEG above about 20 Hz is heavily contaminated by scalp muscle activity<Cite n={3} />, and because band power also rises when the background spectrum flattens, with no rhythm at all<Cite n={4} />. The project asks which explanation fits.</p>
      </Section>

      <Section kicker="2. Spectral estimation" title={<>Time series to <em>power spectral density</em></>}>
        <p>EEG is voltage sampled over time. The Fourier transform splits it into frequencies. The <b>power spectral density (PSD)</b> shows power at each frequency. Welch's method averages spectra from short windows to reduce noise<Cite n={14} />. This project used 2 second windows and a median average.</p>
        <Step title="Maths: sampling, resolution, Welch">
          <Math_ note="Highest measurable frequency. Data were resampled to 512 Hz so a fit up to 120 Hz is safe.">f<sub>Nyquist</sub> = f<sub>s</sub> / 2 = 512 / 2 = 256 Hz</Math_>
          <Math_ note="Frequency resolution. Window length is 1024 samples = 2 s.">Δf = f<sub>s</sub> / N<sub>fft</sub> = 512 / 1024 = 0.5 Hz</Math_>
          <Math_ note="Welch estimate with a median instead of a mean, so brief bursts do not dominate.">P̂(f) = median over windows w of |FFT(x<sub>w</sub> · taper)|² / (f<sub>s</sub> · Σ taper²)</Math_>
        </Step>
      </Section>

      <Section kicker="3. Spectral model" title={<>Periodic and <em>aperiodic</em> components</>}>
        <p>A <b>peak</b> is a bump above the background: a real rhythm at one frequency. The <b>aperiodic</b> background is a smooth downward slope with an exponent (steepness) and an offset (height). specparam models a spectrum as exactly this sum<Cite n={4} />.</p>
        <Math_ note="b = offset, χ = exponent, and each peak k has centre c, height h, width w.">log<sub>10</sub> P(f) = b − χ · log<sub>10</sub> f + Σ<sub>k</sub> h<sub>k</sub> · exp( −(f − c<sub>k</sub>)² / 2w<sub>k</sub>² )</Math_>
        <p>Try it. The shaded strip is the 60 to 110 Hz band measured in meditation studies. Alpha (10 Hz) is always present, as in real eyes-closed EEG.</p>
        <PsdExplorer />
      </Section>

      <Section kicker="4. Confound" title={<>Band power <em>decomposition</em></>}>
        <p>Band power adds up everything in the band, so a peak and a flatter slope can produce the same number. Use the two preset buttons above: both raise ΔB, but only one has a periodic part.</p>
        <Math_ note="Specparam reports the two parts separately, which classic band power cannot.">B = (1 / |F|) Σ<sub>f∈F</sub> log<sub>10</sub> P(f) = B<sub>aperiodic</sub> + B<sub>periodic</sub>, F = 60 to 110 Hz</Math_>
        <Math_ note="Pivot at 10 Hz. Example: χ from 1.5 to 1.0 gives ΔB = 0.5 × 0.92 = 0.46, about ×2.9.">ΔB<sub>aperiodic</sub> = −Δχ · ( L̄ − 1 ), L̄ = mean log<sub>10</sub> f over F ≈ 1.92</Math_>
        <p>The simulation in this project shows it: a pure slope change moved naive band power by Cohen's d = +3.04 while periodic power moved by 0.00<Cite n={15} />.</p>
      </Section>

      <Section kicker="5. Interpretation" title={<>Aperiodic exponent and <em>E/I balance</em></>}>
        <p>Modelling work links a flatter high-frequency slope to relatively more excitation versus inhibition (E/I balance)<Cite n={5} />. The slope also flattens with age<Cite n={6} /> and steepens with loss of consciousness and anaesthesia<Cite n={7} /><Cite n={8} />. Fitting choices such as frequency range and noise can bias the exponent<Cite n={11} />, and muscle activity also flattens it. So the exponent is treated as a candidate marker worth investigating, not a direct readout of E/I balance.</p>
      </Section>

      <Section kicker="6. Methods" title={<>Analysis <em>pipeline</em></>}>
        <p>Each box gives the file, the reasoning, and the maths or settings. All decisions live in <code>configs/default.yaml</code>.<Cite n={15} /></p>
        <Step title="0. Simulate (00_simulate.py, simulate.py): test the method on known truth">
          <p>Why: before trusting a method on real data, check it recovers effects you planted. Five worlds were built (no change, true gamma peak, slope shift, offset shift, both), 40 simulated subjects per scenario (paired: a thinking and a meditation spectrum each), noise 0.02, alpha and beta peaks included, seed 2026.</p>
          <Math_>y = log<sub>10</sub> P(f) + ε, ε ~ N(0, 0.02²)</Math_>
        </Step>
        <Step title="1. Download (01_download.py, data.py): 98 participants, 94 analysed">
          <p>Four 10 minute eyes-closed blocks per person (meditation, thinking, meditation, thinking). Four participants in the undocumented "tm" group were excluded, leaving 63 meditators and 31 controls<Cite n={2} />.</p>
        </Step>
        <Step title="2. Preprocess (02_preprocess.py, preprocess.py): remove non-brain signal">
          <p>Why: blinks, muscle, heartbeat and mains hum would contaminate the spectrum. Steps: resample to 512 Hz, 1 Hz high-pass, average reference, interpolate bad channels, ICA with ICLabel to remove artifact components<Cite n={10} />, cut 2 s epochs. The ICA model uses the <a href="https://mne.tools/mne-icalabel/dev/api/iclabel.html" target="_blank" rel="noopener noreferrer">extended infomax ICA decomposition algorithm</a>, which ICLabel requires.</p>
          <Math_ note="More than 16 bad channels rejects the block. Epochs above 150 µV after ICA are dropped; at least 30 epochs are required.">bad channel if |z<sub>robust</sub>(log var)| &gt; 4 or median peak-to-peak &gt; 300 µV</Math_>
          <Math_ note="ICLabel probability threshold. Muscle proxy: 70 to 200 Hz power at T7, T8, FT7, FT8, TP7, TP8 before ICA.">remove IC if P(class ∈ {"{"}eye, muscle, heart, line, channel noise{"}"}) ≥ 0.80</Math_>
        </Step>
        <Step title="3. Spectra (03_spectra.py, spectra.py): Welch PSD and IRASA">
          <p>Why: the spectrum is the raw material. Line noise at 50, 100, 150, 200 Hz and equipment lines (80, 112, 144 Hz and so on) are interpolated in log space instead of notch filtered, so no hole is cut in the data. IRASA gives an independent slope estimate without fitting peaks<Cite n={9} />.</p>
          <Math_ note="Rhythms shift in frequency under resampling and cancel. The scale-free background stays put.">P<sub>frac</sub>(f) = median<sub>h</sub> √( P(h·f) · P(f / h) ), h ∈ {"{"}1.1, 1.15, …, 1.45{"}"}</Math_>
        </Step>
        <Step title="4. Fit (04_parameterize.py, parameterize.py): specparam">
          <p>Why: this separates rhythm from background. Primary fit 30 to 120 Hz (narrower ranges cut off a broad gamma peak and biased the exponent by about −0.25 in simulation). Peaks must be 2 to 30 Hz wide, at least 0.10 log units high, threshold 2. Peaks pinned at the width limit are not counted.</p>
          <Math_ note="20 fits failed this check. Three outputs per spectrum: naive, periodic, and aperiodic gamma power.">R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)², keep fit if R² ≥ 0.90</Math_>
        </Step>
        <Step title="5. Statistics (05_stats.py, stats.py): state, trait, interaction">
          <p>Why: each person has several blocks, so a mixed model gives each person their own baseline. The interaction tests whether meditation changes the spectrum differently in meditators and controls.</p>
          <Math_ note={<>u<sub>i</sub> ~ N(0, σ²<sub>u</sub>) is the per-person random intercept.</>}>y<sub>ij</sub> = β<sub>0</sub> + β<sub>1</sub>Trait<sub>i</sub> + β<sub>2</sub>State<sub>ij</sub> + β<sub>3</sub>(Trait × State) + β<sub>4</sub>Order + u<sub>i</sub> + ε<sub>ij</sub></Math_>
          <Math_ note="Cohen's d, the effect size used throughout.">d = (x̄<sub>1</sub> − x̄<sub>2</sub>) / s<sub>pooled</sub></Math_>
          <Math_ note={<>Equivalence testing<Cite n={12} />. Lets you argue an effect is small, which an ordinary non-significant test cannot.</>}>TOST: reject H₀ (|Δ| ≥ δ) if p<sub>lower</sub> &lt; .05 and p<sub>upper</sub> &lt; .05, δ = 0.5 SD</Math_>
          <Math_ note="Applied in stats.adjust_pvalues to each model term across the four fitted primary measures (naive, aperiodic gamma power, exponent, offset), alongside Holm. IRASA measures are cross-checks and sit outside the family.">p<sub>(i)</sub> × m / i ≤ q (Benjamini-Hochberg FDR)<Cite n={13} /></Math_>
        </Step>
        <Step title="6. Report and figures (06_figures.py, 08_report.py): numbers come from the tables">
          <p>The report is generated from the committed CSV tables, so text and data cannot drift apart. A 96-specification grid (fit range, knee, method, reference, ICA on or off) shows whether results hold across reasonable choices.</p>
        </Step>
      </Section>

      <Section kicker="7. Validation" title={<>Ground truth <em>simulation</em></>}>
        <p>Ground truth means the known right answer. In real EEG you never know whether a gamma rhythm is present. In a simulation you build the spectra yourself, so you can score the method. Effect sizes below are paired Cohen's d (mean difference divided by the SD of the differences) from the repository's report<Cite n={15} />.</p>
        <Table
          head={["Planted change", "Naive band power", "Periodic power", "Exponent", "Offset"]}
          rows={[
            ["Nothing", "−0.05", "0.00", "+0.03", "−0.06"],
            ["Slope flattened by 0.25", <b>+3.04</b>, <b>0.00</b>, "−3.34", "−0.01"],
            ["Offset raised by 0.15", "+1.14", "−0.16", "−0.05", "+2.34"],
            ["True 80 Hz peak", "+0.61", "+37.51", "−0.06", "−0.02"],
            ["Peak and slope", "+3.28", "+43.91", "−3.15", "−0.01"],
          ]}
        />
        <h3>Simulation settings (configs/default.yaml)</h3>
        <Table
          head={["Parameter", "Value"]}
          rows={[
            ["Simulated subjects per scenario", "40, paired (thinking and meditation spectrum each)"],
            ["Frequency range and step", "3 to 200 Hz, 0.5 Hz"],
            ["Noise level", "0.02"],
            ["Base background", "offset 1.0, exponent 1.5"],
            ["Between-subject SD", "offset 0.15, exponent 0.20"],
            ["Within-subject SD (block drift)", "offset 0.05, exponent 0.05"],
            ["Alpha peak [centre, height, SD]", "[10 Hz, 0.6, 2.0]"],
            ["Beta peak", "[20 Hz, 0.2, 3.0]"],
            ["Planted gamma peak", "[80 Hz, 0.25, 8.0], centre jittered by N(0, 2²)"],
            ["Planted shifts", "offset +0.15, exponent −0.25"],
            ["Random seed", "2026"],
            ["Fit applied to every spectrum", "specparam, 30 to 120 Hz, fixed mode (same as real data)"],
          ]}
        />
        <h3>Planted versus recovered</h3>
        <p>Mean difference (meditation minus thinking) over 40 simulated subjects, in log10 units. The planted exponent shift (−0.25) came back as −0.259 and the planted offset shift (+0.15) as +0.148.<Cite n={15} /></p>
        <Table
          head={["Scenario", "Naive gamma", "Periodic gamma", "Exponent", "Offset", "Peak detected"]}
          rows={[
            ["No change", "−0.008", "0.000", "+0.002", "−0.004", "0% to 0%"],
            ["True 80 Hz peak", "+0.103", <b>+0.093</b>, "−0.005", "−0.002", "0% to 100%"],
            ["Offset +0.15", "+0.154", "0.000", "−0.003", <b>+0.148</b>, "2.5% to 0%"],
            ["Exponent −0.25", "+0.498", <b>0.000</b>, <b>−0.259</b>, "0.000", "0% to 0%"],
            ["Peak and exponent", "+0.581", "+0.093", "−0.254", "−0.001", "0% to 100%"],
          ]}
        />
        <h3>Try the simulated spectra</h3>
        <p>These curves use the exact base parameters above. Pick a planted change and compare the thinking spectrum (dashed) with the meditation spectrum (solid).</p>
        <SimExplorer />
        <p style={{ marginTop: 20 }}><b>What this shows:</b> the method assigns an effect to the right component, with no false alarms. <b>What it does not show:</b> detection of a small peak in noisy data. Simulated effects (d from 3 to 44) dwarf the real ones, and real counted peaks (power 0.09) are weaker than the planted 0.25.</p>
      </Section>

      <Section kicker="8. Results" title={<>Peak detection and <em>exponent</em> effects</>}>
        <div className="facts">
          <div><p className="big">94</p><p>participants analysed: 63 meditators, 31 controls. 340 usable blocks.<Cite n={15} /></p></div>
          <div><p className="big">6%</p><p>of blocks (not participants) had a countable 60 to 110 Hz peak: 21 of 340. Another 20 pinned peaks were excluded.</p></div>
          <div><p className="big">0.96</p><p>correlation between specparam and IRASA exponents. Mean R² of fits: 0.97.</p></div>
          <div><p className="big">p=.013</p><p>state × trait interaction on the exponent, β = −0.109 (95% CI −0.195 to −0.023), uncorrected. FDR-adjusted p = .027.</p></div>
        </div>
        <p>In controls the exponent was steeper during meditation than thinking (+0.099, p = .042). In meditators it barely moved (−0.017, p = .59). The trait main effect was not significant (−0.118, p = .114). Adjusting for muscle proxies did not weaken the trait effect.</p>
        <p><b>Multiple comparisons:</b> correcting each term across the four fitted primary measures, the interaction gives p = .027 with Benjamini-Hochberg FDR but p = .052 with Holm. The interaction therefore holds under FDR control only.<Cite n={13} /></p>
        <p><b>Equivalence:</b> periodic gamma power was equivalent between groups within ±0.5 SD (p = .041). Exponent and offset equivalence were not established.</p>
        <p><b>Limits:</b> one dataset, preliminary; residual muscle activity cannot be fully excluded in a 30 to 120 Hz fit; the exponent is not a pure E/I readout.</p>
      </Section>

      <section>
        <p className="kicker">References</p>
        <h2><em>References</em></h2>
        <ol className="refs">
          {REFERENCES.map((r, i) => <li key={i} id={`r${i + 1}`}>{r}</li>)}
        </ol>
      </section>

      <footer className="site">
        <div>
          Ananya Pandey. Poster presented at CuttingGardens 2026, IIT Mandi. Code: <a href={REPO}>github.com/cation21/Gamma</a> (MIT). Data: <a href="https://openneuro.org/datasets/ds003969">OpenNeuro ds003969</a> (CC0).
        </div>
      </footer>
    </>
  );
}
