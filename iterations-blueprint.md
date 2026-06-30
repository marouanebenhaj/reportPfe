# Section 4 — Iterations (Markdown draft)

> Complete prose draft of the rewritten Section 4, structured as four DSR iterations,
> each following the engineering cycle: **Problem Investigation → Solution Design →
> Design Validation → Design Implementation → Solution Evaluation**.
>
> Inline placeholders mark where existing LaTeX floats are reused:
> `[FIG: label]`, `[TABLE: label]`, `[EQ: label]`. All labels are preserved from the
> current `4-iterations.tex`. Translate each `##`/`###` heading to the matching
> `\subsection`/`\subsubsection`. Prose avoids em dashes per project style.

---

## Section opening

The development of the synthesizer parameter estimation system followed a systematic,
iteration-based process guided by the DSR framework introduced in Section
(`sec:methodology`). Four successive iterations were conducted. The first establishes
the foundational artifact on which every later iteration depends: the data generation
pipeline that produces the labelled dataset. The remaining three iterations form a
refinement chain over the predictive model, advancing from a single-scale baseline
(V1), to a multi-scale architecture with pitch conditioning (V2), and finally to a
multi-pitch data regime with spectral augmentation (V3). Each iteration corresponds to
a complete engineering cycle in which an artifact is investigated, designed, validated,
implemented, and evaluated, with the empirical findings of one iteration feeding
directly into the problem investigation of the next.

[FIG: fig:iterations — high-level diagram of the four iterations and their dependency
chain. NOTE: this figure does not yet exist; flag as optional new asset, mirrors the
reference's "Project Iterations" figure.]

## Key Concepts

> Rationale: some reviewers read the iterations, results, and conclusion without the
> background. As the audience includes software engineers who may not know the audio
> terminology, this short box defines the minimum needed to follow the four iterations.
> Each term is treated in full in the Background (Section [REF: sec:background]); the
> audio-representation terms in particular are detailed in Section
> [REF: ssub:audio_representations].
>
> LaTeX note: render as a single shaded callout box (e.g. a `tcolorbox` or a framed
> `description` list) placed immediately after the section opening. The box and the
> inline pointers in Iteration 1 (Steps 1 and 5) depend on three labels that do not yet
> exist in `2-background-and-literature-review.tex`; add them during the LaTeX step:
> `\label{sec:background}` after `\section{BACKGROUND AND LITERATURE REVIEW}`,
> `\label{ssub:sound_synthesis}` after `\subsubsection{Sound Synthesis and the Vital
> Synthesizer}`, and `\label{ssub:audio_representations}` after
> `\subsubsection{Audio Representations}`. With hyperref loaded, `Section~\ref{...}` is
> automatically clickable.

- **Pitch** — how high or low a note is perceived; physically, the fundamental frequency
  of the tone.
- **MIDI note** — the integer code a keyboard sends for a musical pitch (0 to 127, where
  60 is middle C). The model receives this value as an auxiliary input alongside the
  spectrogram.
- **Timbre** — the "tone colour" that distinguishes two sounds of the same pitch and
  loudness, for example why a bass and a pad played at the same note sound different. The
  five timbre categories (bass, lead, pad, percs, random) organise the dataset.
- **Oscillator** — the component that generates the raw waveform; Vital has two wavetable
  oscillators plus a noise/sample oscillator.
- **Wavetable synthesizer** — a synthesizer (such as Vital) whose oscillator scans
  through a stored table of waveforms; the "wave-frame" position selects the timbre.
- **Envelope (attack / decay)** — how a value such as loudness or filter cutoff evolves
  over time after a note starts; attack is the time to rise, decay the time to fall.
- **Filter (cutoff / resonance)** — shapes timbre by attenuating frequencies above a
  cutoff point; resonance emphasises frequencies near that cutoff.
- **Spectrogram** — a two-dimensional image of a sound, time on one axis and frequency on
  the other, with brightness showing energy. It is what the network sees instead of the
  raw waveform.
- **Mel spectrogram** — a spectrogram whose frequency axis is warped to the mel scale,
  which approximates human pitch perception. This is the model's actual input.
- **Harmonic series** — the frequencies at integer multiples of the fundamental that give
  a pitched sound its structure; resolving them is what the coarse mel scale is best at.

---

# Iteration 1 — Data Pipeline: Constructing the Vital Parameter-Inversion Dataset

The first iteration produces the labelled dataset that all subsequent model iterations
consume. Unlike the later iterations, it does not train or compare models; its artifact
is the data generation pipeline itself and the corpus it yields. It therefore plays the
same foundational role that a threat-modelling framework plays in a security study: it
defines the ground on which everything else is built.

## 1.1 Problem Investigation

Supervised inversion of synthesizer parameters requires a corpus of audio clips paired
with the exact parameter vectors that produced them. No such corpus exists for Vital.
General-purpose audio datasets such as NSynth provide labelled instrument tones but
carry no synthesizer parameters, and the few parameter-labelled datasets in the
literature, used by InverSynth and Sound2Synth, target frequency-modulation
synthesizers whose parameter semantics differ fundamentally from a wavetable
instrument. Building a dataset specific to Vital is therefore a prerequisite rather
than a convenience.

The naive route, sampling the sixteen-dimensional parameter space uniformly at random,
fails in practice. The overwhelming majority of uniform draws produce sounds that are
silent, textureless, or otherwise pathological: both oscillators muted, a filter closed
to inaudibility, or an envelope so short that no sustained tone is ever heard. Such
clips are musically meaningless and waste the substantial compute spent rendering them.
A useful corpus must instead concentrate sampling on the regions of the parameter space
that correspond to recognisable, musically valid sounds.

From these observations, a usable dataset must satisfy five requirements. It must (i)
contain audible, musically coherent sounds spanning a range of timbres; (ii) cover the
musical pitch range so the model is not biased toward a single register; (iii) record
ground-truth parameters that respect Vital's internal quantization, so the targets are
values the synthesizer can actually realise; (iv) admit train, validation, and test
splits that are free of leakage; and (v) present each clip in a representation suited to
a convolutional or transformer backbone. A purpose-built generation pipeline is
therefore required, and its design is the subject of the next phase.

## 1.2 Solution Design

The data generation pipeline transforms raw synthesizer parameter configurations into a
normalized, split dataset ready for model training. It consists of six sequential
steps, illustrated in the pipeline figure. Each step is designed to be checkpoint-aware
and resumable, since the full pipeline involves several hours of audio rendering and
processing.

[FIG: fig:pipeline — the six-step pipeline from parameter extraction to normalized
dataset splits.]

### Step 1 — Parameter Extraction from Vital

*The synthesis concepts used throughout this iteration (oscillators, wavetables,
envelopes, filters, and the MIDI pitch convention) are explained in full in Section
[REF: ssub:sound_synthesis].*

All sixteen target parameters are enumerated programmatically from Vital via the
Pedalboard library, which exposes the synthesizer's internal parameter table. For each
parameter the extraction records its name, internal plugin index, valid range, and
discretization information where applicable. Although all parameters are exposed with
continuous-looking ranges, Vital internally quantizes seven of them to a finite set of
valid discrete values:

- **Envelope times** (`env1_attack`, `env1_decay`, `env2_attack`, `env2_decay`) are
  each mapped to 1,001 discrete steps on a logarithmic time grid.
- **Filter cutoff** (`filter1_cutoff`), expressed in semitones, is quantized to 1,001
  discrete frequency values.
- **Unison voice count** (`osc1_unison_voices`) is inherently discrete, since a
  fractional number of voices has no physical meaning.
- **Oscillator 2 transpose** (`osc2_transpose`) is discrete by musical convention,
  taking integer semitone values.

The extracted ranges and discrete value tables are stored as JSON reference files and
used throughout the pipeline.

[TABLE: tab:parameters — all sixteen parameters in prediction order with discrete status
and description. Sustain and release excluded (unobservable in a 4 s render with no
note-off).]

### Step 2 — Timbre-Constrained Preset Generation

To ensure the dataset covers a representative range of practically meaningful sounds, a
timbre-constrained generation strategy is employed. Five timbre categories are defined,
each associated with a set of parameter sub-ranges that constrain sampling to musically
coherent regions of the parameter space.

[TABLE: tab:timbre_categories — bass / lead / pad / percs / random, with MIDI pitch
ranges, preset counts, and target sound characteristics (44,000 total).]

Each category is governed by a dedicated JSON range file that narrows the sampling
bounds of the sixteen parameters. Within those bounds parameter values are drawn at
random, so the resulting sounds vary widely while still belonging recognisably to the
target category. The `percs` category, for example, caps the amplitude attack at 10 ms,
constrains the decay to a moderate range, and keeps the noise/sample oscillator level
high, guaranteeing a percussive hit regardless of the remaining values. Each preset is
appended to a per-timbre JSON Lines file.

### Step 3 — Audio Rendering

Each preset is rendered to a 44.1 kHz mono WAV file via Pedalboard, which loads Vital as
a VST3 plugin, sets all sixteen parameters programmatically, and triggers a MIDI
note-on event with no corresponding note-off. The clip duration is fixed at 4 seconds.
The absence of a note-off means the release stage of the amplitude envelope is never
triggered, which is why release is excluded from the studied parameters; sustain is
similarly excluded, as the envelope decay is set long enough relative to the clip to
produce effectively sustained sounds without an explicit sustain parameter.

In V1 and V2 each preset is rendered at a single MIDI pitch sampled from the timbre's
range. In V3 each preset is re-rendered at up to three fixed pitches per timbre (see
Iteration 4); the rendering and checkpoint mechanics are described under Design
Implementation.

### Step 4 — Silence Filtering

Certain extreme parameter combinations cause Vital to produce no audible output. These
silent clips are detected and removed using a hybrid energy-based rule applied to each
rendered WAV file.

A straightforward approach would compute the mean energy of the full clip and discard
anything below a fixed threshold. This fails for percussive and transient-heavy sounds:
a stab, a drum hit, or a pluck concentrates nearly all of its energy in the first
10–100 ms, leaving the remaining seconds close to silence, so the clip-level mean falls
below any reasonable threshold even when the sound is perfectly audible. The detector
must therefore distinguish a clip that is uniformly quiet from one that is quiet on
average but contains a brief burst of high energy. The hybrid rule combines an
activity-based criterion with a peak-based criterion.

The signal is divided into short frames of length L with hop H, and the RMS energy of
each frame is computed and converted to decibels.

[EQ: eq:rms — frame RMS energy.]
[EQ: eq:rms_db — conversion to dB.]

Two detection quantities follow: the active ratio R (fraction of frames exceeding the
RMS threshold) and the peak measure P (the maximum of the highest frame RMS and the peak
sample amplitude, both in dB).

[EQ: eq:active_ratio — active ratio R.]
[EQ: eq:peak_db — peak measure P.]

A clip is retained if either condition is satisfied, so transient-heavy sounds that
would fail the active-ratio rule alone are not discarded.

[EQ: eq:keep_rule — keep ⇔ (R > ρ_min) ∨ (P > θ_peak).]
[TABLE: tab:silence_thresholds — per-timbre thresholds θ_rms, ρ_min, θ_peak.]

### Step 5 — Mel Spectrogram Extraction

*The short-time Fourier transform, spectrogram, and mel scale used below are introduced
and derived in full in Section [REF: ssub:audio_representations].*

Each surviving clip is converted to a mel spectrogram. The STFT is computed with a Hann
window, the power spectrogram is taken, and a bank of 128 triangular mel filters between
20 Hz and 12 kHz is applied and converted to decibels.

[EQ: eq:stft — STFT.]
[EQ: eq:power_spec — power spectrogram.]
[EQ: eq:mel — log-mel spectrogram, 128 bins.]

**V1, single-scale.** A single mel spectrogram is extracted using the mid-scale FFT
configuration and stored as a `.npy` file.

**V2, multi-scale.** Three mel spectrograms are extracted per clip at fine, mid, and
coarse resolutions. As the FFT window grows, spectral resolution improves at the cost of
temporal resolution.

[TABLE: tab:mel_scales — fine/mid/coarse n_fft, hop, frame counts, and what each
captures.]
[FIG: fig:multiscale_mel_example — the three scales of a Vital bass note (MIDI 36).]

The three spectrograms are stacked along the channel dimension into a `(3, 128, 224)`
tensor, directly compatible with the three-channel stems of ConvNeXt and ViT backbones.

[FIG: fig:multiscale_stack — fine/mid/coarse stacked into a 3-channel tensor.]

### Step 6 — Dataset Splitting and Normalization

**Stratified split.** A purely random partition would not guarantee that each set
reflects the full musical diversity of the dataset; by chance an entire pitch register
or timbre could be under-represented in validation, so reported MAE would partly reflect
a distribution mismatch rather than genuine generalisation. Each sample is assigned a
bucket key combining its timbre and the octave of its MIDI note, and samples within each
bucket are allocated 80/10/10.

[EQ: eq:bucket — bucket = (timbre, ⌊midi/12⌋).]

For V3, where each preset is rendered at multiple pitches, a sample-level split would
leak: two renders of a preset could land in train and a third in test while sharing the
same target. The V3 split therefore operates at the preset level, grouping all renders
of a preset by the MD5 hash of its sixteen parameter values and assigning the whole
group to one partition.

[EQ: eq:preset_key — k(θ) = MD5(θ_0 … θ_15).]
[FIG: fig:preset_split — sample-level vs preset-level splitting.]

**Normalization.** Three schemes are applied, each fitted only on the training
partition. Mel spectrograms are Z-score normalized per bin; in V2/V3 a separate scaler
is fitted per scale.

[EQ: eq:zscore — per-bin Z-score.]

Twelve of the parameters are min-max normalized linearly, while the four envelope time
parameters are normalized on a logarithmic scale because envelope times span three
orders of magnitude and scale perceptually with the logarithm.

[EQ: eq:minmax — linear min-max.]
[EQ: eq:lognorm — log-scale normalization for envelope times.]

MIDI pitch is normalized to [0, 1] over the fixed range [12, 84] and then passed through
a sinusoidal embedding producing a 128-dimensional Fourier feature vector.

[EQ: eq:pitch_norm — pitch normalization.]
[EQ: eq:sinusoidal — sinusoidal pitch embedding φ(p).]

## 1.3 Design Validation

Before committing to the full multi-hour rendering run, the pipeline design was checked
against the five requirements set out in the problem investigation.

**Audibility (internal validity).** The timbre-constrained ranges were designed so that
any draw within a category's bounds yields an audible sound of that category; the
percussive constraints in particular force a non-trivial onset. The hybrid silence rule
was designed to preserve transient and percussive clips that a mean-energy rule would
wrongly discard, so that legitimate short-onset sounds survive filtering. The discrete
snapping in Step 1 ensures every recorded target is a value Vital can actually realise,
so the model is never asked to predict an unrealisable parameter.

**Leakage freedom (external validity).** The preset-level grouping in Step 6
(`eq:preset_key`) guarantees that every render of a given preset is confined to a single
partition. A post-split check confirmed zero preset-identifier overlap between the
training and test partitions, so evaluation reflects generalisation to unseen presets
rather than to unseen pitch renderings of known presets. The stratified bucket key
(`eq:bucket`) was verified to place every timbre and every pitch register into all three
partitions in proportional numbers, so the evaluation sets are musically representative
of the training distribution.

**Perceptual fidelity of targets.** Logarithmic normalization of the envelope times
(`eq:lognorm`) was adopted because a 10 ms and a 100 ms attack are as perceptually
distinct as a 100 ms and a 1,000 ms attack; equal spacing in log space therefore
represents equal perceptual steps better than equal spacing in linear time, keeping the
regression target aligned with perceived differences.

**Trade-offs.** V1 and V2 use five stratified folds to control split variance at the
~42k-sample scale. V3 instead uses a single preset-level split, because training two
backbone families on ~136k samples for forty epochs five times over is impractical
within the hardware budget, and because the larger corpus substantially reduces the
split variance that motivated cross-validation in the first place. This is a conscious
exchange of repeated-split robustness for feasible compute at scale.

## 1.4 Design Implementation

The pipeline was executed with Pedalboard hosting Vital as a VST3 plugin. Audio
rendering and silence filtering are checkpoint-aware: a hash of each parameter vector,
and for V3 a hash keyed on the (sorted parameters, MIDI note) pair, is checked against
previously written entries so an interrupted run resumes without producing duplicates.
Rendered audio paths and ground-truth parameters are written in append mode to a
combined JSON Lines file, one record per clip, with diagnostic silence fields appended
to each surviving record. Mel spectrograms are stored as `.npy` arrays, and per-split,
per-scale normalization statistics are stored as compressed `.npz` files and reloaded at
evaluation time so that identical normalization is applied to validation and test. The
full generation and feature-extraction run was performed on a single workstation with an
NVIDIA RTX 4090.

## 1.5 Solution Evaluation

The pipeline produced a corpus suitable for the model iterations. The final dataset
sizes after silence filtering and splitting are summarized below.

[TABLE: tab:dataset_sizes — V1/V2 (~42,250 samples, one pitch per preset) and V3
(~168,000 samples, three or four pitches per preset), with train/val/test counts. Splits
performed at preset level in all versions.]

Silence filtering removed the small fraction of presets that rendered to inaudible
output, with the per-timbre thresholds tuned so that percussive clips were retained.
The resulting corpus spans all five timbre categories and the full 12–84 MIDI range in
every partition, and every target respects Vital's internal quantization. With a
leak-free, musically representative, model-ready dataset in place, the central open
question becomes which network architecture best inverts these sounds into their
parameters. That question is the subject of Iteration 2.

---

# Iteration 2 — V1: Baseline Architecture and Pitch Conditioning

This iteration establishes a controlled baseline by training and comparing three
backbone families, a custom residual CNN, an AST, and a ConvNeXt, across three pitch
conditioning strategies, using a single-scale mel spectrogram as input. The goal is to
determine which combination achieves the lowest MAE on the Vital parameter estimation
task and to identify the most promising directions for subsequent refinement.

## 2.1 Problem Investigation

Parameter estimation for the Vital wavetable synthesizer had not been addressed prior to
this study. Bruford et al. applied an AST to Massive, a wavetable synthesizer by Native
Instruments, while InverSynth and Sound2Synth target FM synthesizers. Although Vital is
a wavetable instrument, the sixteen-parameter subset studied here is structured to
replicate a classic subtractive signal chain: two wavetable oscillators and one noise
oscillator feed a low-pass filter whose cutoff is modulated by a dedicated filter
envelope, and the result passes through an amplitude envelope and a global reverb stage.
Despite this subtractive topology, the wavetable oscillators introduce a continuous,
high-dimensional waveform space absent from conventional subtractive or FM instruments.
It is therefore unknown whether the inductive bias of convolutional models, which
capture local spectral patterns, outperforms that of transformer models, which model
long-range dependencies across the full time-frequency plane, when the target combines a
structured analog signal chain with a wavetable stage.

A second concern is explicit pitch conditioning. No prior work on synthesizer parameter
estimation provides the MIDI pitch as a direct model input. Although the fundamental
frequency is implicitly present in the spectrogram as the position of the harmonic
series, it is unclear whether supplying pitch explicitly helps in practice, and there is
reason to expect the effect may be limited. The backbones used here are large (roughly
2M to 28M parameters) and produce high-dimensional mel embeddings on the order of several
hundred dimensions, whereas the pitch enters as a single normalised scalar expanded into
a small embedding. When this small pitch vector is concatenated onto a much larger mel
embedding, the network has an easy path to ignore it and rely on the spectrogram alone.
The tendency is reinforced by the single-pitch data regime of V1: because each preset is
rendered at only one pitch, the same parameter target is never observed at different
pitches, so the model is given no incentive to treat pitch as a disambiguating cue. This
iteration therefore examines whether pitch conditioning yields a measurable improvement
despite these pressures, and whether a richer sinusoidal encoding fares better than a
basic linear projection.

## 2.2 Solution Design

The V1 design comprises model configurations arising from the combination of the
backbone families with three pitch conditioning strategies. All configurations share the
same input, a single mid-scale mel spectrogram of shape 128 × 224, and the same output,
a sigmoid-activated linear projection to sixteen normalised parameters.

The backbones differ in inductive bias, pre-training, and capacity. The SynthParamNet
baseline is a custom residual network of approximately 2M parameters with a hybrid
pooling strategy: the frequency axis is collapsed by global average pooling, and the
resulting time sequence is reduced by concatenating a max-pool and an average-pool,
yielding a 2C-dimensional embedding that handles variable-length time axes without
zero-padding. The AST-small backbone is a pre-trained ViT-small/16 (approximately 22M
parameters) adapted to mono spectrogram input by retaining the first-channel slice of
the RGB stem, with the CLS token as the global representation. The ConvNeXt-tiny backbone
is a pre-trained convolutional network (approximately 28M parameters) with ImageNet-22k
weights, adapted to mono input in the same way and pooled with the same hybrid strategy
as the baseline.

For each backbone, three pitch conditioning variants are evaluated: no-pitch, in which
the mel embedding feeds directly to the head; basic MLP, in which the normalised MIDI
scalar passes through a two-layer MLP and is concatenated with the mel embedding; and
sinusoidal embedding, in which the scalar is first mapped to a 128-dimensional Fourier
feature vector before the same MLP.

[FIG: fig:v1_design — three backbone options × three pitch conditioning strategies,
converging on a shared concatenation fusion and flat regression head.]

## 2.3 Design Validation

Before full-scale training the V1 design was checked for correctness.

- **Architecture compatibility.** The single-channel stem adaptation was verified for
  both AST and ConvNeXt by confirming the adapted first layer produced activations of
  the correct spatial shape, and for AST the dynamic image-size flag was confirmed to
  handle the non-square 128 × 224 input without positional-embedding mismatch.
- **Sinusoidal embedding numerical safety.** The log-spaced frequencies were verified to
  remain within float32 range at the highest index; because they enter only as arguments
  to sine and cosine, no overflow occurs in the embedding.
- **Baseline calibration against prior work.** The AST-small no-pitch configuration was
  designed to approximate the essential setup of Bruford et al., a ViT encoder with
  single-scale mel input and no pitch conditioning, to serve as a reference point. The
  replication is intentionally partial, since Bruford et al. use an MSE loss and scale
  to far larger data; the configuration is therefore a reference rather than a strict
  reproduction.
- **Data split integrity.** The five-fold stratified splits were verified to share no
  preset identifier between any training and test partition.

These checks confirmed the configurations were correctly implemented and that the
AST-small no-pitch variant would serve as a valid reference against prior work.

## 2.4 Design Implementation

**Single-scale mel extraction.** Each clip is converted to a mid-scale mel spectrogram
(n_fft = 2,048, hop = 512, 128 bins) and stored as `.npy`. At training time each
spectrogram is padded with its minimum value or truncated to 128 × 224 and Z-score
normalised using training-split statistics.

**Backbone adaptation to mono input.** For SynthParamNet, a convolutional stem feeds N
residual stages, and the final feature map is reduced to a 2C-dimensional embedding by
frequency-axis average pooling followed by time-axis max-pool ⊕ average-pool. For
AST-small, the first patch-embedding convolution is adapted by retaining the first
channel's weights, and the CLS token serves as the embedding. For ConvNeXt-tiny, the
same single-channel slice is used at the stem with the same hybrid pooling.

**Pitch conditioning variants.** Given the normalised pitch, the no-pitch branch passes
the mel embedding straight to the head; the basic-MLP branch passes the scalar through a
two-layer MLP and concatenates a 64-dimensional pitch vector with the mel embedding; the
sinusoidal branch first encodes the scalar with φ(p) (`eq:sinusoidal`) before the same
MLP.

**Flat regression head.** All six configurations share a three-layer head applied to the
fused embedding, with a final sigmoid bounding predictions to [0, 1].

[EQ: eq:flat_head — three-layer sigmoid head.]

**Training protocol.** Models are trained with AdamW (weight decay 1e-4). Pre-trained
backbones use a differential learning rate, updating the backbone at 0.1× the head rate.
A OneCycleLR schedule with pct_start = 0.3 warms up over the first 30% of training. AMP
and gradient clipping at max-norm 1.0 are used on an RTX 4090. The training loss is an
unweighted SmoothL1 with β = 0.02.

**Five-fold cross-validation.** Five stratified splits are generated from the ~42,250-
sample dataset using the timbre × octave bucket key; for each fold the checkpoint with
the lowest validation MAE is retained, and test MAE is reported as mean ± standard
deviation across folds.

## 2.5 Solution Evaluation

Three patterns emerge from the five-fold results. First, ConvNeXt-tiny outperforms
AST-small in every pitch condition, with the gap ranging from 0.0007 to 0.0017 MAE,
indicating that at the ~42k-sample scale the local convolutional bias extracts more
discriminative timbral features than global self-attention. Second, pitch conditioning
consistently reduces MAE for both backbones; the effect is most pronounced for
`osc2_transpose`, whose error in the no-pitch ConvNeXt configuration is 0.101 compared
to 0.090 with sinusoidal conditioning. Third, the sinusoidal embedding provides a small
but consistent improvement over the basic MLP for both backbones, supporting the
hypothesis that a multi-frequency encoding represents pitch more effectively than a
single linear projection.

Across all configurations the seven oscillator parameters consistently show the highest
per-parameter MAE, typically 0.08 to 0.16, while envelope times and filter parameters
fall below 0.05. This difficulty gradient reflects that wavetable position and unison
settings map non-linearly to the spectrogram, whereas envelope shape is directly visible
in the amplitude trajectory. The best V1 model reaches 0.0758 ± 0.0005 MAE.

These findings establish two directions for the next iteration. The single mid-scale
spectrogram sacrifices either temporal resolution for transient capture or spectral
resolution for harmonic analysis, suggesting that a multi-resolution input could supply
complementary information and reduce oscillator error. In addition, concatenation-based
conditioning adds pitch as a side input but does not let it modulate the features the
backbone has already extracted, limiting its effectiveness. Iteration 3 therefore
introduces a multi-scale three-channel input and replaces concatenation with FiLM.

---

# Iteration 3 — V2: Multi-Scale Input, FiLM, Grouped Head, Weighted Loss

This iteration introduces four enhancements motivated directly by the V1 evaluation: a
multi-scale three-channel mel input, FiLM pitch conditioning, a grouped prediction head,
and a group-weighted loss. The V1 backbone comparison is retained with AST-small and
ConvNeXt-tiny, now trained with the full V2 architecture.

## 3.1 Problem Investigation

The V1 evaluation exposed three concrete limitations of the baseline design.

First, the single mid-scale mel spectrogram (n_fft = 2,048, hop = 512) fixes one
time-frequency trade-off for the entire task. As established in Section
[REF: ssub:audio_representations], no single analysis window resolves both fine temporal
and fine spectral detail at once. The oscillator parameters, which dominate V1 error,
depend precisely on harmonic detail, since discriminating wavetable positions requires
observing how individual harmonics are shaped and spaced, information a single mid window
resolves only partially, while the complementary short-window and long-window views are
simply unavailable to the V1 model.

Second, V1 supplied pitch by concatenating a pitch vector onto the mel embedding only
after the backbone had finished processing the spectrogram. The backbone therefore
extracted its features with no awareness of pitch, and the pitch signal could influence
only the final regression layers rather than the representation itself. The per-parameter
V1 results show this most plainly on `osc2_transpose`, the parameter most directly tied
to pitch, which remained among the harder oscillator parameters even though pitch was
nominally available to the model.

Third, V1 treated all sixteen parameters identically, both in model capacity and in the
training objective. A single flat head and an unweighted loss assigned the same capacity
and the same weight to the easy envelope and filter parameters as to the oscillator
group, even though the per-parameter analysis showed oscillator MAE two to three times
higher. The baseline had no mechanism to concentrate either modelling capacity or
training signal where the error was largest.

## 3.2 Solution Design

Each limitation is addressed by one targeted design change. The concepts introduced in
this iteration are summarised in the box below; FiLM in particular is treated in full in
the Background (Section [REF: ssub:film_conditioning]).

> **Concepts used in this iteration.** LaTeX note: render as a short shaded callout, in
> the same style as the section-opening Key Concepts box. Depends on a new label
> `\label{ssub:film_conditioning}` to be added after the `\subsubsubsection{FiLM
> Conditioning}` heading in `2-background-and-literature-review.tex`.
>
> - **Multi-scale mel input** *(see Section [REF: ssub:audio_representations])* — the
>   three-channel stack of fine, mid, and coarse mel spectrograms, and the time-frequency
>   resolution trade-off that motivates it, are introduced in the Background.
> - **FiLM (Feature-wise Linear Modulation)** *(see Section [REF: ssub:film_conditioning])*
>   — the pitch-conditioning mechanism, in which an auxiliary signal predicts a per-feature
>   scale and shift applied to a feature vector, is defined in the Background.
> - **Grouped prediction head** — a prediction head split into several small sub-networks,
>   one per semantically related group of parameters, instead of one shared MLP for all
>   sixteen outputs. Specific to this work.
> - **Group-weighted loss** — a training objective that scales the loss contribution of
>   each parameter group, so harder groups (here the oscillator parameters) can be
>   emphasised relative to the rest. Specific to this work.

The four changes map directly onto the three limitations:

- **Multi-scale three-channel input** (addresses the fixed time-frequency trade-off). The
  single mid-scale spectrogram is replaced by a stack of three spectrograms at fine, mid,
  and coarse resolutions: the coarse scale resolves harmonic structure, the fine scale
  captures transients and onsets, and the mid scale retains the V1 reference view. The
  three are stacked along the channel dimension into a `(3, 128, 224)` tensor
  (`fig:multiscale_stack`), directly compatible with the three-channel stems of the
  pre-trained ConvNeXt and ViT backbones.
- **FiLM pitch conditioning** (addresses late concatenation of pitch). Concatenation is
  replaced by FiLM: the pitch embedding predicts a scale and shift that are applied to
  the backbone output as an affine transform, so pitch can suppress or amplify individual
  feature dimensions rather than merely appending new ones.
- **Grouped prediction head** (addresses uniform capacity across parameters). The flat
  MLP is replaced by four specialised sub-MLPs, one per parameter group (oscillator,
  envelope, filter, misc), with the oscillator group given a wider sub-network so the
  hardest parameters receive the most capacity.
- **Group-weighted loss** (addresses uniform weighting across parameters). The unweighted
  objective is replaced by a group-weighted SmoothL1 in which the oscillator group's loss
  contribution is doubled, concentrating the training signal on the highest-error
  parameters.

The backbone comparison is held fixed at AST-small and ConvNeXt-tiny so that any change
in MAE is attributable to the four enhancements rather than to a change of architecture.
The FiLM conditioning, grouped head, and weighted loss are formalised under Design
Implementation.

[FIG: fig:v2_design — the V2 (and V3) architecture: 3-channel mel → backbone → FiLM
modulation from the sinusoidal pitch embedding → grouped head → 16 parameters.]

## 3.3 Design Validation

The following checks were performed before full-scale training.

- **Three-channel weight adaptation.** The adapted first-layer weights were verified to
  preserve activation magnitudes: averaging the three RGB slices and dividing by three
  ensures a spectrogram tensor in the ImageNet value range produces activations of the
  same expected magnitude as the original stem, avoiding any learning-rate rescaling.
- **FiLM identity initialisation.** The FiLM MLP was initialised to output zero scale and
  shift, so the modulated embedding equals the unmodulated embedding at epoch zero and
  training begins from a state equivalent to the V1 no-pitch model, avoiding early
  instability.
- **Grouped head output consistency.** The four sub-MLP outputs were verified to
  concatenate in the correct parameter order (indices 0–6, 7–10, 11–12, 13–15), and the
  final sigmoid was confirmed to bound all outputs to [0, 1].
- **Weighted loss gradient balance.** Loss logging confirmed that the doubled oscillator
  weight produces proportionally larger gradients for indices 0–6 without destabilising
  the gradient magnitudes of the remaining groups.

## 3.4 Design Implementation

**Three-channel input construction.** The fine, mid, and coarse spectrograms are each
padded or truncated to 128 × 224 and Z-score normalised independently using
training-split statistics, then stacked along the channel dimension.

**Backbone adaptation to three channels.** Both backbones are adapted from RGB to audio
three-channel input by averaging the pre-trained stem weights across the colour
dimension and scaling.

[EQ: eq:weight_adapt — averaged three-channel stem weights.]

**FiLM pitch conditioning.** A two-layer MLP predicts the scale and shift from the
sinusoidal pitch embedding, and the modulated embedding is formed by an affine
transform.

[EQ: eq:film_pred — (γ, β) = MLP(φ(p)).]
[EQ: eq:film_apply — z̃ = (1 + γ) ⊙ z + β.]

**Grouped prediction head.** The modulated embedding feeds four sub-MLPs, one per group,
whose outputs are concatenated and passed through a sigmoid.

[TABLE: tab:grouped_head — group composition and hidden widths (oscillator at 2h).]
[EQ: eq:grouped_head — per-group sigmoid sub-MLP.]

**Weighted SmoothL1 objective.** The training loss sums per-group SmoothL1 terms with
the oscillator group at double weight, justified because oscillator parameters dominate
timbral character and are the hardest to predict.

[EQ: eq:weighted_loss — group-weighted SmoothL1, w_osc = 2.]

**Training protocol.** All V2 models are trained for forty epochs per fold with an
architecture-dependent batch size (128 or 256). All other settings are inherited from
V1: AdamW (wd 1e-4), differential learning rate, OneCycleLR (pct_start 0.3), AMP, and
gradient clipping at 1.0.

**Ablation design.** To isolate FiLM, each backbone is trained in two variants, the full
V2 model and a no-pitch variant with the FiLM branch removed entirely, so any MAE
difference is attributable solely to pitch conditioning.

## 3.5 Solution Evaluation

The combined V2 enhancements produce a substantial reduction in MAE. The best V2 model,
ConvNeXt-tiny with FiLM at 0.0665 ± 0.0002, improves over the V1 best (0.0758 ± 0.0005)
by roughly 12%, confirming that the three-channel input captures complementary
time-frequency information unavailable to a single scale. ConvNeXt continues to
outperform AST, now by 0.0048 MAE under FiLM.

FiLM provides a small but consistent improvement in the overall average, −0.0016 MAE for
ConvNeXt and −0.0014 for AST, consistent across all five splits. The effect is, however,
sharply concentrated: it falls almost entirely on `osc2_transpose`, the parameter most
directly tied to pitch, which improves from 0.078 to 0.062 under FiLM for ConvNeXt, the
largest single-parameter gain of any change in this iteration. Because `osc2_transpose`
is one parameter among sixteen, this large local gain is diluted into a modest movement
of the overall average, while the remaining parameters are largely unaffected by pitch
conditioning. In other words, on single-pitch data FiLM sharpens exactly the one
parameter for which pitch is a near-direct cue but does little elsewhere.

Both V2 models also show clear overfitting: training MAE falls well below validation MAE,
which plateaus around 0.055 to 0.060 by epoch 30, indicating the ~42,250-sample dataset
is insufficient to fully regularise models of this capacity. The combination of limited
FiLM benefit and overfitting raises the central question for the next iteration: is FiLM
inherently weak for this task, or does its effectiveness depend on training data that
presents the same parameter vector at multiple pitches, forcing the model to use pitch
to resolve spectral ambiguity? Iteration 4 addresses both weaknesses by re-rendering the
existing presets at multiple fixed pitches and adding spectral augmentation.

---

# Iteration 4 — V3: Multi-Pitch Expansion and SpecAugment

This iteration addresses the two weaknesses identified in V2, limited FiLM effectiveness
and mild overfitting, through changes to the training data regime rather than the model
architecture. The V2 architecture (`fig:v2_design`) is retained without modification;
the intervention re-renders each preset at multiple MIDI pitches and adds SpecAugment.

## 4.1 Problem Investigation

The V2 evaluation surfaced three concerns that point to the training data regime rather
than the model architecture. First, both V2 backbones showed clear signs of overfitting.
The quantity optimised during training is the weighted SmoothL1 (Huber) loss, while the
metric tracked on the validation set is the mean absolute error; over the V2 run the two
curves diverged. The training loss kept falling steadily toward zero (below roughly 0.015
by the final epochs) while the validation MAE stopped improving and plateaued around 0.055
to 0.060. This divergence, the model continuing to fit the training set without any
further gain on held-out presets, is the signature of partial memorisation of
preset-specific spectral patterns rather than a generalised mapping. With only ~42,250
training samples shared across backbones of 28M and 23M parameters, regularisation through
greater data diversity is the natural remedy.

Second, the muted FiLM benefit in V2 left the pitch-conditioning question unresolved: it
was unclear whether the mechanism itself is weak or whether the single-pitch data simply
gave it nothing to learn from. In V1 and V2 every preset is rendered at exactly one pitch,
so the model never observes the same parameter vector at different spectral positions;
each training sample is a unique (spectrogram, parameters) pair. In this setting a model
can minimise training loss by associating absolute spectral patterns with parameter
values, using the pitch-dependent position of harmonics as an implicit shortcut, so the
FiLM pathway receives no pressure to modulate features by pitch because pitch-independent
spectral matching already suffices. Rendering each preset at several pitches is therefore
needed to give the FiLM pathway a genuine training signal and to test it properly.

Third, the V2 ranking placed the AST backbone behind ConvNeXt, which is consistent with
Transformers being known to be data-hungry: self-attention has a weaker inductive bias
than convolution and typically needs more data before it pays off. A substantially larger
dataset is needed to see whether AST can realise its potential at this task.

These three concerns motivate two coordinated changes. The dataset is expanded by
re-rendering each preset at three distinct pitches, which simultaneously diversifies the
training distribution (addressing overfitting), supplies the cross-pitch variation the
FiLM pathway requires, and provides the larger data scale that Transformers benefit from.
In addition, SpecAugment is introduced as an online regulariser so that the resulting
models generalise better and their evaluation can be trusted as a reflection of genuine
learning rather than memorisation.

## 4.2 Solution Design

Both interventions in this iteration introduce concepts not used before, and the
evaluation relies on a perceptual metric defined for the first time here. They are
summarised in the box below before the design is described.

> **Concepts used in this iteration.** LaTeX note: render as a short shaded callout, in
> the same style as the section-opening Key Concepts box and the Iteration 3 concepts box.
> Depends on a new label `\label{ssub:specaugment}` to be added after the
> `\subsubsubsection{SpecAugment}` heading in `2-background-and-literature-review.tex`, and
> reuses `\label{eq:mss}` already defined for the MSS equation in the Methodology section.
>
> - **Multi-pitch rendering** — the V3 data-regime change in which each preset is rendered
>   at several fixed MIDI pitches instead of one, so the same parameter vector appears
>   paired with several spectrally distinct clips. Specific to this work.
> - **SpecAugment** *(see Section [REF: ssub:specaugment])* — a spectrogram augmentation
>   that randomly blanks out strips along the time and frequency axes of the input during
>   training, acting as "dropout for spectrograms" to discourage over-reliance on any
>   single time slice or frequency band. Introduced in the Background.
> - **MSS (Multi-Scale Spectral loss)** *(see Section [REF: eq:mss])* — a perceptual
>   evaluation metric that re-renders the predicted parameters through Vital and measures
>   the spectral distance between the original and re-rendered audio across several FFT
>   window sizes. It complements the parameter-space MAE by asking whether predictions
>   actually sound close, and it is used for evaluation only, never as a training loss.

The V3 solution redesigns the data regime. Each of the ~42,250 presets is re-rendered at
up to three fixed pitches per timbre, chosen to cover the low, mid, and high registers of
each timbre's range, so that each parameter vector now appears paired with multiple
spectrally distinct clips. This supplies the FiLM pathway with genuine cross-pitch
signal: it must learn pitch-specific modulations that correct the same backbone embedding
for pitch-dependent spectral shift.

SpecAugment is introduced as an online training-time regulariser, applying time and
frequency masks to each training sample per forward pass to prevent over-reliance on
narrow spectral or temporal features.

[FIG: fig:v3_design — each preset rendered at three fixed pitches yielding three clips
that share one parameter target, each converted to a 3-channel mel with SpecAugment
applied online during training.]

Because each preset now contributes multiple samples, the split must be performed at the
preset level so that all renders of a preset share a partition, a stricter requirement
than the V1/V2 splits enforced through preset-level grouping before stratification.

## 4.3 Design Validation

- **Preset-level split integrity.** The procedure was verified to assign all pitch
  renderings of every preset to the same partition, and a post-split check confirmed
  zero preset-identifier overlap between training and validation, so validation MAE
  reflects generalisation to unseen presets rather than unseen pitches of known presets.
- **SpecAugment parameter selection.** Time masks were capped at 30 frames and frequency
  masks at 15 of 128 bins, chosen so each masked spectrogram retains enough content to
  identify the dominant timbre while providing enough stochastic variation to prevent
  feature co-adaptation.
- **MIDI register coverage.** The fixed pitch schedule per timbre was verified to span
  the low, mid, and high registers of each timbre's range, ensuring at least two octaves
  of pitch variation per class during training.
- **Shared evaluation protocol.** A fixed set of 100 held-out samples was constructed
  from the V1/V2 test set so that V2 and V3 models are evaluated on identical clips and
  targets, enabling a direct MSS comparison without confounding evaluation differences.

## 4.4 Design Implementation

**Multi-pitch rendering.** Each preset is re-rendered at up to three fixed pitches per
timbre following the schedule below, with new renders appended to the existing dataset.
A hash on the (sorted parameters, MIDI note) pair skips any combination already present
from V1/V2: if the V1/V2 random note coincides with a fixed note, only the remaining two
are rendered (three clips total); otherwise all three are rendered (four clips total),
which is typical. After silence filtering the corpus reaches ~168,000 samples.

[TABLE: tab:v3_midi — fixed MIDI pitches per timbre.]

**SpecAugment.** Each training tensor is augmented with two time masks (width drawn from
U[0, 30] frames) and two frequency masks (width from U[0, 15] bins), applied identically
across the three channel slices to preserve inter-channel consistency, with masked
values set to 0.0, which is approximately the mean of the Z-scored input. SpecAugment is
disabled at validation and test time.

**Preset-level split.** The ~42,250 presets are stratified by timbre and split 80/10/10
at the preset level, all renders following their preset's assignment, giving ~134,400
training and ~16,800 validation and test samples. A single split is used rather than
five-fold, owing to the cost of training two backbones on this corpus for forty epochs.

**Training configuration.** The architecture and hyperparameters are identical to V2;
SpecAugment is active only on the training partition.

**MSS evaluation protocol.** Both V2 and V3 models are evaluated with an MSS loss
complementing parameter MAE, but on a deliberately small shared set of only 100 held-out
samples because the MSS pipeline is expensive. Computing MSS for one sample is not a
simple tensor operation: the model's normalised prediction must be denormalised, each
value mapped back to its physical unit (for example semitones or seconds), and then
snapped to a valid setting since most of the studied parameters are discrete in Vital;
the resulting preset is rendered through Vital via Pedalboard, the output audio recorded,
and only then is the mean absolute difference of log-magnitude spectrograms computed
across four FFT scales (128, 512, 2,048, 8,192) against the original. Each sample
therefore triggers a full Vital re-render, which is heavy in both time and compute, so the
evaluation set was capped at 100 samples to keep the MSS sweep over all V2 and V3 models
tractable. The same 100 samples are reused for every model so MAE and MSS are directly
comparable across versions. MSS is never optimised during training.

**Ablation.** The FiLM versus no-pitch comparison from V2 is repeated under the V3
regime, yielding four models; the change in the FiLM gap from V2 to V3 is the primary
test of the multi-pitch hypothesis.

## 4.5 Solution Evaluation

This phase reports only the headline outcome of the iteration; the full results, tables,
and per-parameter and perceptual analyses are presented in the Results and Findings
section (Section 5), all computed on the shared 100-sample evaluation set described above.

Multi-pitch rendering and SpecAugment together produce the largest iteration-over-
iteration improvement in the study: V3 performs substantially better than V2 across every
model variant, confirming that data scale and pitch diversity, not the architecture, were
the dominant bottleneck. The best V3 model is AST-small with FiLM, which moves the
ConvNeXt-versus-AST ranking: the AST backbone, behind ConvNeXt in V1 and V2, overtakes it
in V3, consistent with the expectation that the data-hungry Transformer benefits most from
the enlarged dataset. The FiLM pathway also becomes more effective under multi-pitch data,
and the same trend holds on the perceptual MSS metric. These findings are quantified in
full in Section 5.

Collectively the four iterations answer the research questions posed in the introduction:
Iteration 1 delivers a leak-free, musically representative dataset; Iteration 2
establishes that ConvNeXt outperforms AST at limited data scale; Iteration 3 confirms
that multi-scale input delivers a significant MAE improvement over single-scale; and
Iteration 4 demonstrates that FiLM pitch conditioning requires multi-pitch training data
to function as intended, and that with sufficient data diversity the AST architecture
achieves the best overall parameter estimation and perceptual accuracy.
