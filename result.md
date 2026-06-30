# Section 5 — Results and Findings (Markdown blueprint)

> Detailed plan for the rewritten Results section, mirroring the role
> `iterations-blueprint.md` plays for Section 4. Every number here is verified against
> the real checkpoints and `eval_results/`. Figure and table specs are concrete enough to
> regenerate the assets. Prose avoids em dashes per project style.
>
> Companion data files:
> - `eval_results/summary.csv` — MAE + MSS (4 scales) for all 8 V2/V3 models on the shared 100-sample set
> - `eval_results/per_param_mae.csv` — per-parameter MAE for all 8 V2/V3 models (shared set)
> - `checkpoints1/` (and `synthInvV1/checkpoints/`) — V1 5-fold, 9 configs (incl. baseline CNN)
> - `checkpoints2/` — V2 5-fold, 4 configs (+ per-parameter in `test_results.txt`)

---

## 0. Evaluation protocol (READ FIRST — this is the spine of the section)

Two evaluation regimes exist in this study, and the section must use them consistently:

- **Five-fold CV (sample-level, timbre x octave buckets).** Used for V1 and V2 in
  isolation, where each version is reported on its own protocol. Gives mean +/- std over
  five splits.
- **Shared 100-sample set (`shared_eval_set.json`).** A fixed set of 100 held-out clips on
  which **both V2 and V3** are evaluated for **MAE and MSS**. It is small because MSS is
  expensive: each prediction is denormalised, mapped to physical units, snapped to Vital's
  discrete values, re-rendered through Vital, recorded, and only then scored across four
  FFT scales (see Section 4, Iteration 4, MSS protocol).

**Decision adopted in this blueprint:** every **cross-version (V2 vs V3) comparison uses
the shared 100-sample set for both versions.** This makes `tab:film_gap`, `tab:summary`,
and `tab:per_param` internally consistent (same clips, same targets), and is the only way
MSS can be compared across versions. V1 and V2 standalone tables (`tab:v1_results`,
`tab:v2_results`) keep their 5-fold numbers. One sentence in 5.3 states this explicitly.

**Why this matters (and why it is the stronger story):** on the shared single-pitch set,
V2 ConvNeXt+FiLM (0.07391) is actually *worse* than V2 ConvNeXt no-pitch (0.07210), while
V2 AST+FiLM (0.07553) beats AST no-pitch (0.07832). FiLM is therefore **fragile and
backbone-dependent on single-pitch data**, and V3's multi-pitch regime makes it help
**both** backbones. That fragile-to-robust arc is a cleaner answer to RQ3 than the current
draft, which hides the ConvNeXt reversal by quoting 5-fold V2 MAE.

---

## 1. Verified reference data (source of truth for every table/figure)

### 1.1 V1 — five-fold test MAE (9 configs)
| Backbone | Pitch | MAE (mean +/- std) | Delta vs AST no-pitch |
|---|---|---|---|
| SynthParamNet (CNN) | none | 0.0850 +/- 0.0014 | +0.0057 |
| SynthParamNet (CNN) | basic MLP | 0.0828 +/- 0.0007 | +0.0035 |
| SynthParamNet (CNN) | sinusoidal | 0.0829 +/- 0.0008 | +0.0036 |
| AST-small | none (baseline) | 0.0793 +/- 0.0006 | --- |
| AST-small | basic MLP | 0.0781 +/- 0.0005 | -0.0012 |
| AST-small | sinusoidal | 0.0776 +/- 0.0007 | -0.0017 |
| ConvNeXt-tiny | none | 0.0767 +/- 0.0004 | -0.0026 |
| ConvNeXt-tiny | basic MLP | 0.0765 +/- 0.0007 | -0.0028 |
| **ConvNeXt-tiny** | **sinusoidal** | **0.0758 +/- 0.0005** | **-0.0035** (V1 best) |

ConvNeXt-vs-AST gaps (per pitch mode): none 0.0026, basic 0.0016, sinusoidal 0.0018.
**Range = 0.0016 to 0.0026** (the draft's "0.0009 to 0.0017" is wrong — fix it).

### 1.2 V2 — five-fold test MAE (standalone table)
| Version | Model | MAE (mean +/- std) | Delta vs V1 baseline 0.0793 |
|---|---|---|---|
| V1 | AST-small, no pitch (baseline) | 0.0793 +/- 0.0006 | --- |
| V1 | ConvNeXt-tiny, sinusoidal (V1 best) | 0.0758 +/- 0.0005 | -4.4% |
| V2 | ConvNeXt-tiny, no pitch | 0.0681 +/- 0.0002 | -14.1% |
| V2 | AST-small, no pitch | 0.0727 +/- 0.0003 | -8.3% |
| V2 | AST-small, FiLM | 0.0713 +/- 0.0003 | -10.1% |
| V2 | **ConvNeXt-tiny, FiLM** | **0.0665 +/- 0.0002** | **-16.1%** (V2 best) |

V2 validation plateau (from training logs): ConvNeXt Eval=0.0657, AST Eval=0.0702.
**So "validation MAE plateaus near 0.066 to 0.070"** (Section 4's "0.055 to 0.060" is wrong;
Section 5's "near 0.067" is right). Training **loss** (not MAE) bottoms at ~0.010 to 0.014.

### 1.3 Shared 100-sample set — MAE + MSS (V2 and V3, all 8 models)
| Ver | Backbone | Pitch | MAE | MSS avg | mss8192 | mss2048 | mss512 | mss128 |
|---|---|---|---|---|---|---|---|---|
| V2 | ConvNeXt | FiLM | 0.07391 | 1.40535 | 1.39954 | 1.34982 | 1.37866 | 1.49337 |
| V2 | ConvNeXt | none | 0.07210 | 1.38284 | 1.39028 | 1.33692 | 1.34655 | 1.45760 |
| V2 | AST | FiLM | 0.07553 | 1.29154 | 1.27867 | 1.21803 | 1.26659 | 1.40288 |
| V2 | AST | none | 0.07832 | 1.27321 | 1.27528 | 1.22659 | 1.23015 | 1.36083 |
| V3 | ConvNeXt | FiLM | 0.05296 | 1.22155 | 1.17333 | 1.14480 | 1.21324 | 1.35482 |
| V3 | ConvNeXt | none | 0.05498 | 1.26357 | 1.23096 | 1.19669 | 1.25431 | 1.37232 |
| V3 | **AST** | **FiLM** | **0.05248** | **1.09566** | 1.04299 | 1.01850 | 1.09129 | 1.22987 |
| V3 | AST | none | 0.05596 | 1.11903 | 1.07911 | 1.04749 | 1.10117 | 1.24834 |

**FiLM gap (shared set):**
- V2 ConvNeXt: 0.07391 vs 0.07210 = **+0.0018 (FiLM HURTS, +2.5%)**
- V2 AST: 0.07553 vs 0.07832 = **-0.0028 (-3.6%)**
- V3 ConvNeXt: 0.05296 vs 0.05498 = **-0.0020 (-3.7%)**
- V3 AST: 0.05248 vs 0.05596 = **-0.0035 (-6.2%)**

### 1.4 Per-parameter MAE — V2 best vs V3 best (shared set, consistent)
V2 best = ConvNeXt+FiLM (`V2 ConvNeXt +pitch` column); V3 best = AST+FiLM (`V3 AST +pitch`).
| Group | Parameter | V2 best | V3 best | reduction |
|---|---|---|---|---|
| Oscillator | osc1_wave_frame | 0.1355 | 0.1015 | 25% |
| Oscillator | osc1_level | 0.1041 | 0.0823 | 21% |
| Oscillator | osc1_unison_voices | 0.1058 | 0.1022 | 3% (sticky) |
| Oscillator | osc1_unison_detune | 0.0795 | 0.0681 | 14% |
| Oscillator | osc2_wave_frame | 0.1516 | 0.1205 | 21% (hardest) |
| Oscillator | osc2_level | 0.1175 | 0.0868 | 26% |
| Oscillator | osc2_transpose | 0.0759 | 0.0308 | **59%** (pitch effect) |
| Envelope | env1_attack | 0.0230 | 0.0117 | 49% |
| Envelope | env1_decay | 0.0189 | 0.0103 | 45% |
| Envelope | env2_attack | 0.0406 | 0.0237 | 42% |
| Envelope | env2_decay | 0.0296 | 0.0213 | 28% |
| Filter | filter1_cutoff | 0.0398 | 0.0294 | 26% |
| Filter | filter1_resonance | 0.0873 | 0.0498 | 43% |
| Misc | sample_level | 0.0779 | 0.0457 | 41% |
| Misc | mod1_amount | 0.0358 | 0.0290 | 19% |
| Misc | reverb_mix | 0.0598 | 0.0266 | 56% |

(If a V1 column is wanted it must come from V1 5-fold best split and only has 15 params —
reverb_mix did not exist in V1. Cleanest: keep this table V2-vs-V3 only, mention V1 best
per-parameter in prose.)

### 1.5 Canonical dataset numbers (use these everywhere; fix Section 5)
- V1/V2: ~42,250 samples after silence filtering (1 pitch per preset), 5-fold 80/10/10.
- V3: ~42,250 presets re-rendered at up to 3 fixed pitches → 167,572 generated → **~168,000**
  samples; single preset-level split 133,994 / 16,803 / 16,775.
- **Do not write 42,520 or 170,000** (current Section 5 errors).

---

## 2. Section structure and prose plan

Heading: `\section{RESULTS AND FINDINGS}` `\label{sec:results}`.

### 5.0 Opening paragraph
One paragraph: the section reports results across the three **model** iterations (V1, V2,
V3); the data pipeline (Iteration 1) produced no model results. Map RQ1 -> 5.1, RQ2 -> 5.2,
RQ3 -> 5.3. State the dual protocol in one sentence: V1 and V2 are reported on five-fold
CV; all V2-vs-V3 comparisons use the shared 100-sample set required by MSS.

> Note: keep "three iterations" here (V1/V2/V3) but add a half-sentence that the foundational
> data pipeline is Iteration 1 in Section 4, so it does not read as contradicting the
> four-iteration framing.

### 5.1 RQ1 — Architecture and Pitch Conditioning (V1)  `\label{subsec:rq1}`
**Goal:** which backbone, which pitch encoding, at the 42,250-sample scale.

- **TABLE `tab:v1_results`** — the 9-config table from 1.1 (5-fold, AST-no-pitch as the
  Bruford-style reference, ConvNeXt+sinusoidal in bold). Caption notes the dagger baseline.
- **FIGURE `fig:v1_architecture`** (`figs/fig_v1_architecture.pdf`) — grouped bar chart,
  9 bars (3 backbones x 3 pitch modes), y = five-fold test MAE with error bars, dotted red
  horizontal line at the 0.0793 AST-no-pitch reference. Already exists; OK as-is.
- **Prose — three findings:**
  1. ConvNeXt beats AST in every pitch mode (gaps **0.0016 to 0.0026**). Local convolutional
     bias wins at limited data scale.
  2. The custom CNN trails both pre-trained backbones in all conditions -> ImageNet-22k
     pre-training transfers to mel regression even with 1-channel adaptation.
  3. Explicit pitch helps every backbone. Restate precisely: **pitch (either encoding) beats
     no-pitch in all three backbones; sinusoidal beats basic-MLP in 2 of 3** (it ties/loses on
     the CNN by 0.0001). Drop the muddled "five of six / all six" wording.
- Close: name ConvNeXt-tiny + sinusoidal the V1 champion; osc2_transpose improves most under
  pitch, motivating explicit conditioning in V2.

### 5.2 RQ2 — Multi-Scale Input and Architectural Enhancements (V2)  `\label{subsec:rq2}`
**Goal:** isolate the value of multi-scale mel + grouped head + weighted loss from pitch.

- **TABLE `tab:v2_results`** — table from 1.2 (5-fold; V1 baseline + V1 best + 4 V2 configs).
- **FIGURE `fig:convergence`** (`figs/fig_convergence.pdf`) — two panels of train/val curves
  over epochs. Left: V2 ConvNeXt+FiLM (split 0), shaded train-to-val gap. Right: V3 AST+FiLM,
  visibly smaller gap. **Fix label:** the train curve is the **SmoothL1 loss**, not "training
  MAE"; the val curve is MAE. Caption and any in-figure legend must say "training loss" vs
  "validation MAE."
- **Prose:**
  1. **No-pitch V2 ConvNeXt (0.0681) already beats V1 best (0.0758) by 10.2%** with zero pitch
     info -> attributable purely to multi-scale input + grouped head + weighted loss. This is
     the direct RQ2 answer.
  2. FiLM adds a further -0.0016 (2.4%) ConvNeXt and -0.0014 (1.9%) AST on five-fold; small,
     deferred to RQ3.
  3. Overfitting: training **loss** keeps dropping to ~0.010 to 0.014 while validation **MAE**
     plateaus near **0.066 to 0.070** (ConvNeXt 0.0657, AST 0.0702). The train/val divergence
     (two different metrics, but the divergence is the signal) motivates V3's data expansion.

### 5.3 RQ3 — FiLM Effectiveness and the Role of Multi-Pitch Data  `\label{subsec:rq3}`
**Protocol sentence here:** "To compare V2 and V3 directly, and because MSS requires
re-rendering each prediction through Vital, the remaining results are computed on a fixed
shared set of 100 held-out clips evaluated identically for both versions."

#### 5.3.1 FiLM conditioning gap: V2 vs V3 (shared set)
- **TABLE `tab:film_gap`** — rebuilt on the **shared set** (from 1.3):
  | Ver | Backbone | Pitch | MAE | FiLM gain |
  |---|---|---|---|---|
  | V2 | ConvNeXt | FiLM | 0.07391 | **+0.0018 (FiLM hurts)** |
  | V2 | ConvNeXt | none | 0.07210 | |
  | V2 | AST | FiLM | 0.07553 | -0.0028 (-3.6%) |
  | V2 | AST | none | 0.07832 | |
  | V3 | ConvNeXt | FiLM | 0.05296 | -0.0020 (-3.7%) |
  | V3 | ConvNeXt | none | 0.05498 | |
  | V3 | **AST** | **FiLM** | **0.05248** | **-0.0035 (-6.2%)** |
  | V3 | AST | none | 0.05596 | |
- **FIGURE `fig:film_gap`** (`figs/fig_film_gap.pdf`) — grouped bars, FiLM vs no-pitch per
  backbone in V2 and V3, annotated Delta. **REGENERATE:** the current figure uses 5-fold V2;
  switch the V2 pair to shared-set values so it matches the rebuilt table. The ConvNeXt V2 bar
  pair will now show FiLM slightly *above* no-pitch (the fragility point).
- **Prose (the RQ3 payoff):** on single-pitch V2 data, FiLM is unreliable: it slightly hurts
  ConvNeXt and helps AST. Under V3's multi-pitch regime, where each parameter vector is seen at
  three spectral positions, FiLM helps **both** backbones and by a larger margin (ConvNeXt
  -3.7%, AST -6.2%). This confirms the RQ3 hypothesis: FiLM needs multi-pitch data to function
  as intended. Tie to osc2_transpose (5.3.4).

#### 5.3.2 Cross-iteration summary
- **TABLE `tab:summary`** — headline MAE + MSS, **all shared-set** so each row is internally
  consistent:
  | Ver | Model | Pitch | MAE | MSS | Delta MAE vs V2 ConvNeXt+FiLM 0.07391 |
  |---|---|---|---|---|---|
  | V2 | ConvNeXt | FiLM | 0.07391 | 1.405 | --- |
  | V2 | AST | FiLM | 0.07553 | 1.292 | +2.2% |
  | V3 | ConvNeXt | FiLM | 0.05296 | 1.222 | -28.3% |
  | V3 | **AST** | **FiLM** | **0.05248** | **1.096** | **-29.0%** |
  (Optionally add a V1 reference row from 5-fold, clearly labelled as a different protocol.)
- **Prose:** V3 cuts shared-set MAE by ~29% over V2 and the ConvNeXt->AST ranking flips
  (AST 0.05248 < ConvNeXt 0.05296; AST MSS 1.096 < 1.222). Consistent with Transformers being
  data-hungry: AST overtakes once the corpus is 4x larger.

#### 5.3.3 Perceptual evaluation: Multi-Scale Spectral loss
- **FIGURE `fig:mss_scales`** (`figs/fig_mss_scales.pdf`) — line plot, x = 4 FFT scales
  (128, 512, 2048, 8192), y = MSS, 8 series (V2 dashed, V3 solid; ConvNeXt vs AST coloured).
  Source = shared set (already correct). OK as-is.
- **Prose:** every V3 model beats its V2 counterpart at every FFT scale (use exact per-scale
  numbers from 1.3: V3 AST+FiLM ranges 1.043 at 8192 to 1.230 at 128). Because MSS is computed
  via Vital re-render and never trained on, MAE-MSS agreement shows the gains are audible.
- **Keep the honest V2 MSS wrinkle:** in V2, no-pitch has slightly lower MSS than FiLM
  (ConvNeXt 1.383 vs 1.405; AST 1.273 vs 1.292) — the same single-pitch FiLM fragility seen in
  MAE. It disappears in V3 (FiLM lower MSS on both). This now agrees with the MAE story instead
  of contradicting it (the payoff of using shared-set MAE in 5.3.1).

#### 5.3.4 Per-parameter analysis
- **TABLE `tab:per_param`** — V2-best vs V3-best from 1.4 (both shared set, 16 params, group
  rows). Drop the V1 column (or move V1 best to a sentence) to keep one evaluation set.
- **FIGURE `fig:per_param`** (`figs/fig_per_param.pdf`) — grouped bar chart, 16 params on x,
  V2-best and V3-best bars, background shading per group (osc/env/filter/misc). **REGENERATE**
  if currently 3-series V1/V2/V3 mixed-protocol; make it 2-series V2-vs-V3 shared-set (or, if
  keeping V1, annotate its different protocol and 15-param caveat).
- **Prose:**
  - Oscillator group stays hardest (osc2_wave_frame 0.1205, osc1_wave_frame 0.1015 in V3):
    wavetable position is a continuous index into a high-dimensional waveform space.
  - **osc2_transpose is the headline: 0.0759 -> 0.0308 (59% drop)** — direct evidence of
    multi-pitch + FiLM, since transpose is the parameter most tied to pitch.
  - Envelope params near-ceiling (env1_attack 0.0117), directly observable as the amplitude
    trajectory.
  - osc1_unison_voices stays sticky (0.1022, only ~3% better): a discrete parameter with a
    non-monotonic spectral effect that continuous regression handles poorly. Good limitation
    hook for the Discussion.

---

## 3. Figure inventory (status after the protocol fix)
| Figure | File | Content | Source | Status |
|---|---|---|---|---|
| `fig:v1_architecture` | fig_v1_architecture.pdf | 9-bar V1 MAE chart + baseline line | V1 5-fold | OK |
| `fig:convergence` | fig_convergence.pdf | train-loss vs val-MAE curves, V2 left / V3 right | training logs | Relabel "training loss" (not MAE) |
| `fig:film_gap` | fig_film_gap.pdf | FiLM vs no-pitch bars, V2 + V3 | shared set | **Regenerate** (V2 -> shared) |
| `fig:mss_scales` | fig_mss_scales.pdf | MSS across 4 FFT scales, 8 series | shared set | OK |
| `fig:per_param` | fig_per_param.pdf | per-parameter bars by group | shared (V2/V3) | **Regenerate** to V2-vs-V3 shared |

Plot regeneration is driven by `generate_plots.py`; the source CSVs already hold the
shared-set numbers, so regenerating `fig_film_gap` and `fig_per_param` only requires pointing
them at the shared-set V2 columns.

---

## 4. Critique fixes folded into this blueprint
1. **Protocol consistency** — all V2-vs-V3 comparisons on the shared 100-sample set (0.1).
2. **Dataset numbers** — 42,250 and ~168,000 throughout (was 42,520 / 170,000).
3. **Loss vs MAE** — "training loss" everywhere the SmoothL1 objective is meant (not "training MAE").
4. **Validation plateau** — ~0.066 to 0.070 (and Section 4's 0.055 to 0.060 must be corrected to match).
5. **ConvNeXt-AST gap range** — 0.0016 to 0.0026 (was 0.0009 to 0.0017).
6. **Pitch-mode claim** — precise "pitch beats no-pitch 3/3; sinusoidal beats basic 2/3" (drop "five of six").
7. **Per-parameter table** — single evaluation set (V2-vs-V3 shared), V1 caveated or dropped.

## 5. Open decisions for the user
- **A. Protocol framing:** adopt shared-set-for-cross-version as above (recommended), or keep
  the current 5-fold-V2 vs shared-V3 mix with an explicit caveat. This blueprint assumes the
  former.
- **B. V1 in `tab:per_param` / `fig:per_param`:** drop V1 (cleanest, recommended) or keep it
  with a 15-param + 5-fold caveat.
- **C. V1 reference row in `tab:summary`:** include (labelled as 5-fold) or omit.
