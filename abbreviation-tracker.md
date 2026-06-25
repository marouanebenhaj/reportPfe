# Abbreviation First Occurrence Tracker

Tracks where each abbreviation is first defined (full name written out) in the report body.
Format: defined = full name written at this location. Subsequent uses are abbreviation only.

| Abbreviation | Full Name                                          | Section        | Subsection         | Status   |
|--------------|----------------------------------------------------|----------------|--------------------|----------|
| GA           | Genetic Algorithm                                  | Introduction   | Project Context    | defined  |
| CNN          | Convolutional Neural Network                       | Introduction   | Project Context    | defined  |
| AST          | Audio Spectrogram Transformer                      | Introduction   | Project Context    | defined  |
| mel          | Mel Scale (perceptual frequency scale)             | Introduction   | Project Context    | used only — compound word (mel spectrogram), no inline definition needed |
| ConvNeXt     | Conv. Network in the Style of Vision Transformers  | Introduction   | Purpose of Study   | used only — proper model name, no inline definition needed |
| FiLM         | Feature-wise Linear Modulation                     | Introduction   | Purpose of Study   | defined  |
| MIDI         | Musical Instrument Digital Interface               | Introduction   | Purpose of Study   | defined  |
| MAE          | Mean Absolute Error                                | Introduction   | Scope              | defined  |
| MSS          | Multi-Scale Spectral Loss                          | Introduction   | Scope              | defined  |
| DSR          | Design Science Research                            | Introduction   | Approach & Boundaries | defined (not in abbreviations list — consider adding) |
| WAV          | Waveform Audio File Format                         | Introduction   | Limitations        | defined  |
| VST          | Virtual Studio Technology                          | Introduction   | Delimitations      | defined  |
| MP3          | Moving Picture Experts Group Audio Layer III       | Introduction   | Delimitations      | defined  |
| AMP          | Automatic Mixed Precision                          | Iterations     | V1 Training Protocol | defined  |
| AdamW        | Adaptive Moment Estimation with Weight Decay       | Iterations     | V1 Training Protocol | defined  |
| CLS          | Classification Token (Transformer)                 | Background     | Vision Transformers and AST | defined  |
| DFT          | Discrete Fourier Transform                         | Background     | Audio Representations | defined  |
| dB           | Decibels                                           | Background     | Physics of Sound Generation | defined  |
| DDSP         | Differentiable Digital Signal Processing           | Background     | Differentiable Synthesis | defined  |
| DSP          | Digital Signal Processing                          | TBD            | TBD                | not yet encountered |
| FM           | Frequency Modulation                               | Iterations     | V1 Problem Investigation | defined  |
| FFT          | Fast Fourier Transform                             | Background     | Audio Representations | defined  |
| JSON         | JavaScript Object Notation                         | Iterations     | Step 1: Parameter Extraction | defined  |
| JSONL        | JSON Lines                                         | Iterations     | Step 2: Preset Generation    | defined  |
| LFO          | Low-Frequency Oscillator                           | TBD            | TBD                | not yet encountered |
| LR           | Learning Rate                                      | TBD            | TBD                | not yet encountered |
| MIR          | Music Information Retrieval                        | TBD            | TBD                | not yet encountered |
| MLP          | Multi-Layer Perceptron                             | Background     | FiLM Conditioning  | defined  |
| NFT          | Normalising Flow Transform                         | TBD            | TBD                | not yet encountered |
| ReLU         | Rectified Linear Unit                              | Iterations     | V1 Solution Implementation | defined  |
| RMS          | Root Mean Square                                   | Iterations     | Step 4: Silence Filtering  | defined  |
| SpecAugment  | Spectrogram Augmentation                           | Background     | SpecAugment        | used as proper name (like ConvNeXt) |
| STFT         | Short-Time Fourier Transform                       | Background     | Audio Representations | defined  |
| ViT          | Vision Transformer                                 | Background     | Vision Transformers and AST | defined  |

## Notes
- All introduction abbreviations are now defined at first use. Remaining **TBD** entries will be tracked as their sections are written.
- **DSR** is used inline but not in the abbreviations list — consider adding it.
- Abstract is standalone: abbreviations are defined there independently and do not count as first use for the report body.
