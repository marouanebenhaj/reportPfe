# Neural Inversion of Synthesizer Parameters from Audio
## A Comparative Study of Convolutional and Transformer Architectures

> **Capstone Project — Audio Signal Processing & Deep Learning**
> *Supervisor Review Document*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Formulation](#2-problem-formulation)
3. [Tools & Technologies](#3-tools--technologies)
4. [Study Parameters](#4-study-parameters)
5. [Data Pipeline](#5-data-pipeline)
   - [Step 1: Parameter Extraction](#step-1-parameter-extraction-from-vital)
   - [Step 2: Timbre-Constrained Preset Generation](#step-2-timbre-constrained-preset-generation)
   - [Step 3: Audio Rendering](#step-3-audio-rendering)
   - [Step 4: Silence Filtering](#step-4-silence-filtering)
   - [Step 5: Mel Spectrogram Extraction](#step-5-mel-spectrogram-extraction)
   - [Step 6: Dataset Splitting & Normalization](#step-6-dataset-splitting--normalization)
6. [Model Architectures](#6-model-architectures)
   - [V1: Baseline Models](#v1-baseline-models)
   - [V2: Enhanced Architectures](#v2-enhanced-architectures)
7. [Training Strategy](#7-training-strategy)
8. [Ablation Study Design](#8-ablation-study-design)
9. [V3: Current Work in Progress](#9-v3-current-work-in-progress)
10. [Key Design Decisions & Rationale](#10-key-design-decisions--rationale)
11. [Project Status Summary](#11-project-status-summary)

---

## 1. Project Overview

This project addresses the problem of **synthesizer parameter inversion**: given an audio recording produced by a synthesizer, predict the exact values of the synthesizer parameters that generated it. This is a challenging, under-explored regression task at the intersection of audio signal processing, music information retrieval, and deep learning.

The synthesizer of choice is **Vital**, a modern wavetable synthesizer. The parameter space is deliberately scoped to a curated subset of 16 parameters inspired by the classic analog signal flow of the **Minimoog** and **Prophet-5**, making the task tractable while remaining musically meaningful.

**Core Task:** `Audio Signal → Deep Neural Network → Synthesizer Parameter Vector ∈ [0, 1]^16`

---

## 2. Problem Formulation

### Task Type
Supervised multi-output regression.

### Input
- Mel spectrogram of a rendered audio clip (up to 4 seconds)
- Normalized MIDI pitch value (scalar, 0–1)

### Output
- A 16-dimensional vector of normalized synthesizer parameters, each in `[0, 1]`

### Loss Function
Smooth L1 Loss (`β = 0.02`) — chosen for robustness to outliers in parameter prediction while maintaining gradient smoothness near zero.

### Evaluation Metric
Mean Absolute Error (MAE) — both overall and per-parameter.

---

## 3. Tools & Technologies

| Tool | Role |
|---|---|
| **Vital Synthesizer** | Audio generation engine; source of ground-truth parameters |
| **Pedalboard** (Spotify) | Python plugin host; used to load Vital, set parameters, and render audio programmatically |
| **Librosa** | Audio analysis, mel spectrogram extraction, RMS energy computation |
| **SoundFile** | Low-level WAV file I/O |
| **PyTorch** | Model definition, training, and inference |
| **timm** | Pre-trained vision models (ViT, ConvNeXt) adapted for audio spectrograms |
| **NumPy** | Numerical operations, array manipulation |
| **Python** | Full pipeline orchestration |

---

## 4. Study Parameters

The 16 parameters chosen for this study follow the canonical analog synthesizer signal chain. Design decisions were guided by the Minimoog and Prophet-5 architectures.

```
oscillator_1_wave_frame       — Wavetable position (timbre)
oscillator_1_level            — OSC1 amplitude
oscillator_1_unison_voices    — Number of unison voices
oscillator_1_unison_detune    — Detuning between unison voices
oscillator_2_wave_frame       — Wavetable position
oscillator_2_level            — OSC2 amplitude
oscillator_2_transpose        — Pitch offset (semitones)
sample_level                  — Noise/sample oscillator level
envelope_1_attack             — Amplitude envelope: attack time
envelope_1_decay              — Amplitude envelope: decay time
filter_1_cutoff               — Filter cutoff frequency (semitones)
filter_1_resonance            — Filter resonance
modulation_1_amount           — Filter envelope modulation depth (unipolar: 0→1)
envelope_2_attack             — Filter envelope: attack time
envelope_2_decay              — Filter envelope: decay time
reverb_mix                    — Reverb wet/dry mix
```

### Design Rationale for Parameter Exclusions

**Sustain** was excluded because all audio clips are rendered with no note-off event. By setting a decay time much longer than the 4-second clip duration, sustained sounds are effectively emulated without requiring an explicit sustain parameter.

**Release** was excluded because without a note-off event in the rendered clips, the release stage is never triggered, making it unobservable from the audio.

**Envelope 1** acts as the **amplitude envelope**. **Envelope 2** is the **filter envelope**, and its influence on the filter cutoff is controlled by `modulation_1_amount` (unipolar, 0–1 range only).

### Discrete Parameters in Vital

Vital internally quantizes certain parameters to discrete valid values. The following parameters are treated as discrete during preset generation:

```
envelope_1_attack       — 1001 valid values
envelope_1_decay        — 1001 valid values
envelope_2_attack       — 1001 valid values
envelope_2_decay        — 1001 valid values
filter_1_cutoff         — 1001 valid values
oscillator_1_unison_voices
oscillator_2_transpose
```

During preset generation, continuous random samples are snapped to the nearest valid discrete value using a float-extraction utility that handles unit strings (e.g., `"2.00002e-08 secs"`, `"-51.36 semitones"`).

---

## 5. Data Pipeline

### Step 1: Parameter Extraction from Vital

Using the **Pedalboard** Python library, all of Vital's exposed parameters are enumerated programmatically, extracting:
- Parameter name
- Internal index
- Valid range (`min`, `max`)
- Discretization information (where applicable)

The 16 study parameters and their valid discrete values are extracted and stored as JSON reference files.

---

### Step 2: Timbre-Constrained Preset Generation

Rather than sampling uniformly across all parameter combinations — which would produce mostly random, textureless sounds — a **timbre-aware preset generation** strategy is used.

For each of five timbre categories, a **timbre range file** (JSON) is defined that constrains specific parameters to musically meaningful sub-ranges. This ensures that any random parameter draw within those constraints produces a recognizable sound of the target timbre.

**Timbre Categories & MIDI Ranges:**

| Timbre | Min MIDI | Max MIDI | Presets Generated |
|---|---|---|---|
| `bass` | C0 (12) | C3 (48) | 6,000 |
| `lead` | C3 (48) | C6 (84) | 6,000 |
| `pad` | C2 (36) | C5 (72) | 6,000 |
| `percs` | C1 (24) | C6 (84) | 6,000 |
| `random` | C0 (12) | C6 (84) | 20,000 |

The `random` category covers the full musical range and captures sounds that do not cleanly belong to the other four families.

**Example: `percs` timbre range constraints (excerpt)**

```json
{ "name": "envelope_1_attack",  "min": 0.0,  "max": 0.01, "unit": "secs", "discrete": true },
{ "name": "envelope_1_decay",   "min": 0.05, "max": 0.6,  "unit": "secs", "discrete": true },
{ "name": "sample_level",       "min": 0.5,  "max": 1.0,  "discrete": false }
```

The percussive timbre is enforced by tightening the amplitude envelope (very short attack, moderate decay) and boosting the noise/sample oscillator level.

All generated presets are stored in per-timbre **JSONL files**, with checkpoint-aware deduplication to allow resumable generation runs.

---

### Step 3: Audio Rendering

Each preset is rendered to a WAV file via Pedalboard + Vital. Key aspects:

- Each preset is rendered at one or more MIDI notes (sampled from the timbre's pitch range)
- Clips are up to **4 seconds** in duration
- A hashing mechanism tracks already-rendered presets, enabling safe **resume from checkpoint** using append-mode writing to the combined dataset JSONL
- Output: `combined_dataset.jsonl` with one record per rendered clip, containing the audio file path, preset parameters, MIDI note, and a unique `preset_id`

---

### Step 4: Silence Filtering

A **hybrid silence detection** algorithm filters out clips where the synthesizer produced no audible output (e.g., extreme parameter combinations that mute the signal). The detector uses both:

1. **Active ratio rule**: fraction of RMS frames above an energy threshold must exceed `min_active_ratio`
2. **Peak rule**: maximum RMS dB or sample peak amplitude must exceed `peak_db_threshold`

A clip is **kept** if *either* condition is satisfied (`active_ok OR peak_ok`), ensuring that transient-heavy sounds like percussions — which have high energy in few frames — are not mistakenly discarded.

**Per-timbre thresholds:**

| Timbre | RMS Threshold (dB) | Min Active Ratio | Peak Threshold (dB) |
|---|---|---|---|
| `percs` | -50.0 | 0.002 | -30.0 |
| `bass` | -50.0 | 0.020 | -28.0 |
| `lead` | -50.0 | 0.020 | -28.0 |
| `pad` | -55.0 | 0.015 | -30.0 |
| `random` | -50.0 | 0.010 | -30.0 |

Diagnostic fields (`silence_rms_db`, `silence_active_ratio`, `silence_max_rms_db`, `silence_peak_db`) are appended to each surviving JSONL record.

---

### Step 5: Mel Spectrogram Extraction

#### V1: Single-Scale
Standard mel spectrogram with **128 mel bins** is extracted from each audio clip. The spectrogram is stored as a `.npy` file and referenced in the dataset JSONL.

#### V2: Multi-Scale
Three mel spectrograms are extracted per clip using different FFT/hop window configurations:
- **Fine** scale: high temporal resolution
- **Mid** scale: balanced (used as the legacy single-scale reference)
- **Coarse** scale: high frequency resolution

Each scale is stored separately (`{stem}_fine.npy`, `{stem}_mid.npy`, `{stem}_coarse.npy`).

Additionally, a **harmonic vector** is computed per clip (`{stem}_harm.npy`) and several spectral descriptors are appended to the JSONL row:
`spectral_centroid`, `spectral_flatness`, `norm_rolloff`, `avg_contrast`, `avg_zcr`, `avg_rms`, `attack_time`.

---

### Step 6: Dataset Splitting & Normalization

#### Stratified Split

A **stratified splitting strategy** ensures uniform distribution across the training, validation, and test sets. The stratification key is:

```
bucket_key = timbre × octave(midi_note // 12)
```

Within each bucket, indices are **deterministically shuffled** using a seeded RNG, then split in the ratio **80 / 10 / 10**.

**5-fold cross-validation** is implemented by repeating the split with different random seeds (`seed + split_idx` for `split_idx` in 0–4), producing 5 independent train/eval/test JSONL files per split.

#### Normalization

| Feature | Method | Scope |
|---|---|---|
| Mel spectrograms | Z-score (μ=0, σ=1) | Per-split: scaler fitted on train, applied to eval/test |
| Synthesizer parameters | Min-Max to [0, 1] | Per-split: same scaler scheme |
| MIDI pitch | Min-Max to [0, 1] | Fixed range (12–84) |

**Total dataset size: ~44,000 samples (V1/V2); ~160,000 samples (V3 with pitch augmentation)**

---

## 6. Model Architectures

All models share a common **dual-input design**: a mel spectrogram encoder fused with a pitch encoder, followed by a prediction head.

### V1: Baseline Models

Three encoder variants were implemented and compared:

---

#### 1. Baseline CNN (`SynthParamNet`)

A custom residual convolutional network.

**MelEncoder:**
- Stem: `ConvBNReLU(1 → base_ch)`
- N stages, each containing M `ResidualBlock`s followed by a strided `ConvBNReLU` (doubles channels, halves spatial size)
- **Hybrid pooling**: frequency-axis mean → time-axis (`max` + `avg` concatenated) → embedding of size `out_ch × 2`

**Pitch Module (`PitchMLP`):**
- `SinusoidalPitchEmbedding`: maps scalar MIDI (normalized 0–1) to a 128-dim Fourier feature vector using log-spaced frequencies `[2^0·π, …, 2^10·π]`
- Two-layer MLP over the embedding

**Fusion & Head:**
`[mel_emb || pitch_emb] → Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Linear(16) → Sigmoid`

**Configurations:**

| Preset | mel_base | stages | blocks/stage | fusion_hidden |
|---|---|---|---|---|
| `baseline-tiny` | 32 | 3 | 1 | 128 |
| `baseline-small` | 32 | 3 | 2 | 256 |
| `baseline-medium` | 32 | 4 | 2 | 384 |

---

#### 2. AST — Audio Spectrogram Transformer (`ASTSynthParamNet`)

Uses a pre-trained **Vision Transformer (ViT)** from `timm` as the mel encoder.

- Input conv layer modified from 3-channel (RGB) to 1-channel (mono spectrogram) by taking the first channel's weights
- `dynamic_img_size=True` allows non-square inputs (128×224)
- Output: **CLS token** embedding
- Pre-trained models: `vit_base_patch16_224`, `vit_small_patch16_224`

---

#### 3. ConvNeXt (`ConvNeXtSynthParamNet`)

Uses a pre-trained **ConvNeXt** backbone from `timm`.

- First conv layer adapted from 3-channel to 1-channel (mono spectrogram)
- **Hybrid pooling**: spatial max + spatial average over feature map → `out_ch × 2` embedding
- Pre-trained models: `convnext_tiny_in22k`, `convnext_base_in22k`

---

#### V1 Results Summary

ConvNeXt-tiny outperformed both the baseline CNN and AST-small with and without the sinusoidal pitch module. ConvNeXt emerged as the primary architecture for V2.

---

### V2: Enhanced Architectures

V2 introduces three architectural innovations applied uniformly across all model families:

---

#### Innovation 1: Multi-Scale Input

Three mel spectrograms (fine, mid, coarse) are stacked into a **3-channel input tensor** of shape `(B, 3, 128, 224)`.

For **ConvNeXt and ViT**, the first conv layer is adapted for 3-channel input with weights initialized by averaging the pre-trained RGB weights (divided by 3 to preserve activation magnitudes).

For the **baseline CNN**, three independent `MelEncoder` branches process each scale separately, and their embeddings are concatenated: `out_ch × 2 × 3`.

---

#### Innovation 2: FiLM Pitch Conditioning (`FiLMConditioner`)

Instead of naive concatenation of pitch and mel embeddings, **Feature-wise Linear Modulation (FiLM)** is used:

```
γ, β = MLP(pitch_embedding)
z_conditioned = (1 + γ) ⊙ mel_embedding + β
```

The FiLM MLP is initialized to zero output (identity transform at initialization). This allows the pitch information to multiplicatively and additively modulate the mel feature map, rather than appending it as a side channel.

---

#### Innovation 3: Grouped Prediction Head (`GroupedPredictionHead`)

Instead of a single shared MLP predicting all 16 parameters, four **specialized sub-MLPs** are used — one per semantically coherent parameter group:

| Group | Parameters | Head Width |
|---|---|---|
| Oscillator | 7 params (indices 0–6) | `hidden × 2` (wider — harder task) |
| Envelope | 4 params (indices 7–10) | `hidden // 2` |
| Filter | 2 params (indices 11–12) | `hidden // 2` |
| Misc | 3 params (indices 13–15) | `hidden // 2` |

All group outputs are concatenated and passed through `Sigmoid` to produce the final `[0, 1]` predictions.

---

#### V2 Model Variants

| Preset | Backbone | Multi-Scale | FiLM | Grouped Head |
|---|---|---|---|---|
| `convnext-tiny-v2` | ConvNeXt-Tiny | ✅ | ✅ | ✅ |
| `convnext-base-v2` | ConvNeXt-Base | ✅ | ✅ | ✅ |
| `ast-small-v2` | ViT-Small/16 | ✅ | ✅ | ✅ |
| `ast-base-v2` | ViT-Base/16 | ✅ | ✅ | ✅ |
| `baseline-v2` | Custom CNN | ✅ | ✅ | ✅ |

---

#### V2 Results

V2 achieved approximately **15% MAE improvement** over V1 on the validation set. However, two problems were identified:

1. **Overfitting**: Models still exhibit a train/eval MAE gap, suggesting insufficient regularization or dataset size.
2. **FiLM had minimal impact**: Models with and without FiLM conditioning produced nearly identical results. Hypothesis: the model sees each preset rendered at only a single pitch, so there is no cross-pitch variation to force the model to leverage pitch as a conditioning signal.

---

## 7. Training Strategy

### Optimizer
- **AdamW** with weight decay `1e-4`
- For pre-trained models (AST, ConvNeXt): **differential learning rates** — backbone LR = `lr × 0.1`, head LR = `lr`

### Scheduler
- **OneCycleLR**: cosine annealing with linear warm-up (`pct_start=0.3`)

### Mixed Precision
- `torch.amp.autocast` + `GradScaler` for CUDA training

### Gradient Clipping
- `max_norm = 1.0`

### Spectrogram Collation
- **CNN models**: variable-length time axis, padded to batch maximum with zeros
- **AST/ConvNeXt models**: all spectrograms resized/padded/truncated to exactly `128 × 224` (min-value padding at boundaries)

### Cross-Validation
- 5-fold splits, each trained for 20 epochs with `batch_size = 32`
- Best model per split saved by minimum eval MAE

---

## 8. Ablation Study Design

V2 includes a systematic ablation suite to isolate the contribution of each architectural component:

| Ablation Variant | Change |
|---|---|
| `convnext-tiny-v2-mid` | Single-scale input (mid only, no fine/coarse) |
| `convnext-tiny-v2-nofilm` | Concat pitch instead of FiLM conditioning |
| `convnext-tiny-v2-flathead` | Single flat MLP head instead of grouped heads |
| `convnext-tiny-v2-nopitch` | No pitch input at all |
| `convnext-tiny-v2-unweighted` | Uniform loss weighting (training flag only, same architecture) |

Each variant is controlled — only one factor changes at a time against the full `convnext-tiny-v2` baseline.

---

## 9. V3: Current Work in Progress

### Motivation

The FiLM conditioning failure in V2 was traced to a **data diversity problem**: each preset was rendered at only one pitch per sample. The model therefore never observes the *same preset* at *different pitches*, and has no incentive to learn pitch-dependent feature modulation.

### Solution: Multi-Pitch Rendering

Each V1/V2 preset is re-rendered at **three different MIDI pitches** (sampled from the timbre's valid range). This means:

- The FiLM conditioner now receives the same preset with different spectrograms but identical parameter targets, forcing it to learn pitch-invariant parameter prediction while using pitch to disambiguate ambiguous spectral features
- The dataset grows from **~44,000 → ~160,000 samples**, substantially improving coverage and expected to reduce overfitting

### Expected Outcomes

- Improved FiLM conditioning effectiveness (measurable via the `no-pitch` ablation gap)
- Reduced train/eval MAE gap due to larger dataset
- Improved generalization across the pitch range

> **Status: Models are currently under training. Results are pending.**

---

## 10. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Scoping to 16 parameters | Classic analog signal flow (Minimoog/Prophet-5) gives interpretable, structured parameter space |
| Timbre-constrained preset generation | Ensures uniform coverage of musically meaningful sounds; avoids silent or pathological outputs |
| Discrete parameter snapping | Respects Vital's internal quantization; prevents the model from learning to predict values that are invalid |
| Hybrid RMS silence detection | Pure energy thresholds miss transient percussion; the peak rule ensures short-duration sounds survive |
| Sinusoidal pitch embedding | Avoids float32 overflow from naive power-of-2 scaling; provides rich Fourier feature representation for the pitch scalar |
| Per-split normalization | Prevents data leakage from eval/test statistics into the training scaler |
| Stratified split by timbre × octave | Ensures each timbre is represented at each pitch range in every split |
| Grouped prediction head | Oscillator parameters (waveform, unison) are harder than envelope times; giving the oscillator head more capacity is principled |
| FiLM over concatenation | Multiplicative modulation is more expressive than additive concatenation; init to identity prevents training instability |

---

## 11. Project Status Summary

| Phase | Status |
|---|---|
| Parameter extraction & preset schema | ✅ Complete |
| Timbre range files (bass, lead, pad, percs, random) | ✅ Complete |
| Preset generation (~44k presets) | ✅ Complete |
| Audio rendering pipeline | ✅ Complete |
| Silence filtering | ✅ Complete |
| V1 single-scale mel extraction | ✅ Complete |
| V2 multi-scale mel extraction | ✅ Complete |
| Dataset splitting (5-fold, stratified) | ✅ Complete |
| Normalization pipeline | ✅ Complete |
| V1 model training & comparison | ✅ Complete (ConvNeXt-tiny best) |
| V2 model training & comparison | ✅ Complete (+15% MAE improvement) |
| V2 ablation study | ✅ Complete (FiLM minimal, grouped head helps) |
| V3 multi-pitch dataset generation | ✅ Complete (~160k samples) |
| V3 model training | 🔄 In Progress |
| Final evaluation & analysis | ⏳ Pending |

---

*Document prepared for academic supervision. Last updated: April 2026.*