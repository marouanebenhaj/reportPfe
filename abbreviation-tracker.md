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
| FiLM         | Feature-wise Linear Modulation                     | Introduction   | Purpose of Study   | needs definition added |
| MIDI         | Musical Instrument Digital Interface               | Introduction   | Purpose of Study   | needs definition added |
| MAE          | Mean Absolute Error                                | Introduction   | Scope              | defined  |
| MSS          | Multi-Scale Spectral Loss                          | Introduction   | Scope              | defined  |
| DSR          | Design Science Research                            | Introduction   | Approach & Boundaries | defined (not in abbreviations list — consider adding) |
| WAV          | Waveform Audio File Format                         | Introduction   | Limitations        | needs definition added |
| VST          | Virtual Studio Technology                          | Introduction   | Delimitations      | needs definition added |
| AMP          | Automatic Mixed Precision                          | TBD            | TBD                | not yet encountered |
| AdamW        | Adaptive Moment Estimation with Weight Decay       | TBD            | TBD                | not yet encountered |
| CLS          | Classification Token (Transformer)                 | TBD            | TBD                | not yet encountered |
| DDSP         | Differentiable Digital Signal Processing           | TBD            | TBD                | not yet encountered |
| DSP          | Digital Signal Processing                          | TBD            | TBD                | not yet encountered |
| FFT          | Fast Fourier Transform                             | TBD            | TBD                | not yet encountered |
| LFO          | Low-Frequency Oscillator                           | TBD            | TBD                | not yet encountered |
| LR           | Learning Rate                                      | TBD            | TBD                | not yet encountered |
| MIR          | Music Information Retrieval                        | TBD            | TBD                | not yet encountered |
| MLP          | Multi-Layer Perceptron                             | TBD            | TBD                | not yet encountered |
| NFT          | Normalising Flow Transform                         | TBD            | TBD                | not yet encountered |
| ReLU         | Rectified Linear Unit                              | TBD            | TBD                | not yet encountered |
| RMS          | Root Mean Square                                   | TBD            | TBD                | not yet encountered |
| SpecAugment  | Spectrogram Augmentation                           | TBD            | TBD                | not yet encountered |
| STFT         | Short-Time Fourier Transform                       | TBD            | TBD                | not yet encountered |
| ViT          | Vision Transformer                                 | TBD            | TBD                | not yet encountered |

## Notes
- Abbreviations marked **needs definition added** still appear without their full name at first use in the introduction — these should be fixed next.
- **DSR** is used inline but not in the abbreviations list — consider adding it.
- Abstract is standalone: abbreviations are defined there independently and do not count as first use for the report body.
