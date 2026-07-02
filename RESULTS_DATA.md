# RAW RESULTS DATA (extracted from checkpoints/logs)

Auto-extracted from `C:\Users\Splinter\Documents\pfe`. Train = SmoothL1/Huber loss; Eval/Test = MAE. 'Best epoch' = epoch of lowest eval MAE in that split.

**Directory-name legend.** V1: a plain name (`ast-small`, `convnext-tiny`, `baseline-small`) = **basic-MLP pitch**; `*_no_pitch` = **no pitch**; `*_sin_embed_44k` = **sinusoidal pitch embedding**. V2/V3: `*-v2` = full model with **FiLM**; `*-v2-nopitch` = **no pitch** (FiLM branch removed). `baseline-*` = custom CNN (`SynthParamNet`); `ast-*` = ViT; `convnext-*` = ConvNeXt. V1 has 100 epochs/split, V2 has 40 epochs/split, V3 has 40 epochs (single split).

---
## V1  (single-scale mel, 5-fold CV)

### ast-small

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 92 | 0.0083 | 0.0779 |
| 1 | 90 | 0.0087 | 0.0780 |
| 2 | 87 | 0.0093 | 0.0782 |
| 3 | 84 | 0.0096 | 0.0777 |
| 4 | 92 | 0.0081 | 0.0782 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0779 | 0.0776 | 0.0787 | 0.0787 | 0.0778 | 0.0781 ± 0.0005 | 1 (0.0776) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1376 |
| oscillator_1_level | 0.1187 |
| oscillator_1_unison_voices | 0.1261 |
| oscillator_1_unison_detune | 0.0985 |
| oscillator_2_wave_frame | 0.1632 |
| oscillator_2_level | 0.1353 |
| oscillator_2_transpose | 0.0949 |
| envelope_1_attack | 0.0211 |
| envelope_1_decay | 0.0171 |
| envelope_2_attack | 0.0328 |
| envelope_2_decay | 0.0247 |
| filter_1_cutoff | 0.0410 |
| filter_1_resonance | 0.0706 |
| sample_level | 0.0813 |
| modulation_1_amount | 0.0335 |
| reverb_mix | 0.0458 |


### ast-small_no_pitch

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 84 | 0.0095 | 0.0791 |
| 1 | 84 | 0.0098 | 0.0797 |
| 2 | 91 | 0.0085 | 0.0796 |
| 3 | 84 | 0.0097 | 0.0787 |
| 4 | 82 | 0.0102 | 0.0798 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0786 | 0.0790 | 0.0804 | 0.0796 | 0.0790 | 0.0793 ± 0.0006 | 0 (0.0786) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1410 |
| oscillator_1_level | 0.1182 |
| oscillator_1_unison_voices | 0.1224 |
| oscillator_1_unison_detune | 0.0934 |
| oscillator_2_wave_frame | 0.1627 |
| oscillator_2_level | 0.1347 |
| oscillator_2_transpose | 0.1116 |
| envelope_1_attack | 0.0216 |
| envelope_1_decay | 0.0168 |
| envelope_2_attack | 0.0334 |
| envelope_2_decay | 0.0248 |
| filter_1_cutoff | 0.0397 |
| filter_1_resonance | 0.0728 |
| sample_level | 0.0868 |
| modulation_1_amount | 0.0325 |
| reverb_mix | 0.0460 |


### ast-small_sin_embed_44k

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 87 | 0.0094 | 0.0776 |
| 1 | 84 | 0.0097 | 0.0776 |
| 2 | 84 | 0.0098 | 0.0778 |
| 3 | 83 | 0.0100 | 0.0777 |
| 4 | 86 | 0.0094 | 0.0783 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0775 | 0.0763 | 0.0780 | 0.0783 | 0.0782 | 0.0776 ± 0.0007 | 1 (0.0763) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1384 |
| oscillator_1_level | 0.1169 |
| oscillator_1_unison_voices | 0.1233 |
| oscillator_1_unison_detune | 0.0962 |
| oscillator_2_wave_frame | 0.1581 |
| oscillator_2_level | 0.1334 |
| oscillator_2_transpose | 0.0892 |
| envelope_1_attack | 0.0207 |
| envelope_1_decay | 0.0172 |
| envelope_2_attack | 0.0328 |
| envelope_2_decay | 0.0249 |
| filter_1_cutoff | 0.0409 |
| filter_1_resonance | 0.0708 |
| sample_level | 0.0798 |
| modulation_1_amount | 0.0329 |
| reverb_mix | 0.0447 |


### baseline-small

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0828 | 0.0814 | 0.0833 | 0.0832 | 0.0834 | 0.0828 ± 0.0007 | 1 (0.0814) |

**Per-parameter MAE (best split):** _not available — per-parameter values in the source `test_results.txt` are corrupt (all logged as ~0.000)._


### baseline-small_no_pitch

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 76 | 0.0347 | 0.0855 |
| 1 | 86 | 0.0309 | 0.0833 |
| 2 | 96 | 0.0265 | 0.0842 |
| 3 | 93 | 0.0268 | 0.0861 |
| 4 | 79 | 0.0325 | 0.0858 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0853 | 0.0824 | 0.0862 | 0.0861 | 0.0852 | 0.0850 ± 0.0014 | 1 (0.0824) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1486 |
| oscillator_1_level | 0.1251 |
| oscillator_1_unison_voices | 0.1231 |
| oscillator_1_unison_detune | 0.0918 |
| oscillator_2_wave_frame | 0.1782 |
| oscillator_2_level | 0.1430 |
| oscillator_2_transpose | 0.1136 |
| envelope_1_attack | 0.0214 |
| envelope_1_decay | 0.0151 |
| envelope_2_attack | 0.0367 |
| envelope_2_decay | 0.0282 |
| filter_1_cutoff | 0.0438 |
| filter_1_resonance | 0.0718 |
| sample_level | 0.0867 |
| modulation_1_amount | 0.0398 |
| reverb_mix | 0.0515 |


### baseline-small_sin_embed_44k

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 80 | 0.0333 | 0.0823 |
| 1 | 86 | 0.0319 | 0.0816 |
| 2 | 76 | 0.0361 | 0.0823 |
| 3 | 81 | 0.0326 | 0.0835 |
| 4 | 74 | 0.0363 | 0.0844 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0824 | 0.0818 | 0.0829 | 0.0838 | 0.0837 | 0.0829 ± 0.0008 | 1 (0.0818) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1487 |
| oscillator_1_level | 0.1276 |
| oscillator_1_unison_voices | 0.1286 |
| oscillator_1_unison_detune | 0.0937 |
| oscillator_2_wave_frame | 0.1750 |
| oscillator_2_level | 0.1404 |
| oscillator_2_transpose | 0.0903 |
| envelope_1_attack | 0.0212 |
| envelope_1_decay | 0.0153 |
| envelope_2_attack | 0.0379 |
| envelope_2_decay | 0.0283 |
| filter_1_cutoff | 0.0456 |
| filter_1_resonance | 0.0717 |
| sample_level | 0.0885 |
| modulation_1_amount | 0.0413 |
| reverb_mix | 0.0547 |


### convnext-tiny

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 76 | 0.0135 | 0.0769 |
| 1 | 83 | 0.0115 | 0.0759 |
| 2 | 82 | 0.0120 | 0.0762 |
| 3 | 80 | 0.0127 | 0.0767 |
| 4 | 80 | 0.0123 | 0.0768 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0768 | 0.0752 | 0.0771 | 0.0769 | 0.0764 | 0.0765 ± 0.0007 | 1 (0.0752) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1329 |
| oscillator_1_level | 0.1165 |
| oscillator_1_unison_voices | 0.1083 |
| oscillator_1_unison_detune | 0.0864 |
| oscillator_2_wave_frame | 0.1597 |
| oscillator_2_level | 0.1305 |
| oscillator_2_transpose | 0.0951 |
| envelope_1_attack | 0.0233 |
| envelope_1_decay | 0.0196 |
| envelope_2_attack | 0.0354 |
| envelope_2_decay | 0.0276 |
| filter_1_cutoff | 0.0413 |
| filter_1_resonance | 0.0670 |
| sample_level | 0.0784 |
| modulation_1_amount | 0.0352 |
| reverb_mix | 0.0467 |


### convnext-tiny_no_pitch

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 81 | 0.0125 | 0.0769 |
| 1 | 91 | 0.0105 | 0.0765 |
| 2 | 80 | 0.0130 | 0.0763 |
| 3 | 84 | 0.0120 | 0.0770 |
| 4 | 77 | 0.0129 | 0.0768 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0767 | 0.0761 | 0.0771 | 0.0772 | 0.0764 | 0.0767 ± 0.0004 | 1 (0.0761) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1340 |
| oscillator_1_level | 0.1174 |
| oscillator_1_unison_voices | 0.1093 |
| oscillator_1_unison_detune | 0.0886 |
| oscillator_2_wave_frame | 0.1576 |
| oscillator_2_level | 0.1298 |
| oscillator_2_transpose | 0.1014 |
| envelope_1_attack | 0.0232 |
| envelope_1_decay | 0.0207 |
| envelope_2_attack | 0.0366 |
| envelope_2_decay | 0.0282 |
| filter_1_cutoff | 0.0414 |
| filter_1_resonance | 0.0676 |
| sample_level | 0.0796 |
| modulation_1_amount | 0.0356 |
| reverb_mix | 0.0465 |


### convnext-tiny_sin_embed_44k

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 78 | 0.0132 | 0.0752 |
| 1 | 80 | 0.0131 | 0.0760 |
| 2 | 74 | 0.0140 | 0.0755 |
| 3 | 77 | 0.0134 | 0.0755 |
| 4 | 88 | 0.0112 | 0.0758 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0753 | 0.0753 | 0.0764 | 0.0763 | 0.0755 | 0.0758 ± 0.0005 | 0 (0.0753) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1362 |
| oscillator_1_level | 0.1161 |
| oscillator_1_unison_voices | 0.1067 |
| oscillator_1_unison_detune | 0.0869 |
| oscillator_2_wave_frame | 0.1610 |
| oscillator_2_level | 0.1288 |
| oscillator_2_transpose | 0.0902 |
| envelope_1_attack | 0.0245 |
| envelope_1_decay | 0.0191 |
| envelope_2_attack | 0.0367 |
| envelope_2_decay | 0.0269 |
| filter_1_cutoff | 0.0415 |
| filter_1_resonance | 0.0679 |
| sample_level | 0.0809 |
| modulation_1_amount | 0.0346 |
| reverb_mix | 0.0461 |


---
## V2  (multi-scale + FiLM + grouped head + weighted loss, 5-fold CV)

### ast-base-v2

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 35 | 0.0214 | 0.0697 |
| 1 | 36 | 0.0198 | 0.0693 |
| 2 | 34 | 0.0251 | 0.0706 |
| 3 | 35 | 0.0227 | 0.0703 |
| 4 | 34 | 0.0245 | 0.0694 |


### ast-small-v2

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 36 | 0.0181 | 0.0718 |
| 1 | 36 | 0.0180 | 0.0707 |
| 2 | 35 | 0.0198 | 0.0718 |
| 3 | 34 | 0.0233 | 0.0715 |
| 4 | 37 | 0.0157 | 0.0702 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0713 | 0.0712 | 0.0716 | 0.0714 | 0.0708 | 0.0713 ± 0.0003 | 4 (0.0708) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1281 |
| oscillator_1_level | 0.1013 |
| oscillator_1_unison_voices | 0.1252 |
| oscillator_1_unison_detune | 0.0943 |
| oscillator_2_wave_frame | 0.1581 |
| oscillator_2_level | 0.1145 |
| oscillator_2_transpose | 0.0628 |
| envelope_1_attack | 0.0172 |
| envelope_1_decay | 0.0133 |
| envelope_2_attack | 0.0333 |
| envelope_2_decay | 0.0264 |
| filter_1_cutoff | 0.0405 |
| filter_1_resonance | 0.0713 |
| sample_level | 0.0628 |
| modulation_1_amount | 0.0367 |
| reverb_mix | 0.0475 |


### ast-small-v2-nopitch

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 35 | 0.0198 | 0.0733 |
| 1 | 36 | 0.0178 | 0.0719 |
| 2 | 36 | 0.0179 | 0.0727 |
| 3 | 36 | 0.0178 | 0.0731 |
| 4 | 35 | 0.0213 | 0.0724 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0726 | 0.0725 | 0.0723 | 0.0732 | 0.0729 | 0.0727 ± 0.0003 | 2 (0.0723) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1327 |
| oscillator_1_level | 0.1062 |
| oscillator_1_unison_voices | 0.1247 |
| oscillator_1_unison_detune | 0.0943 |
| oscillator_2_wave_frame | 0.1554 |
| oscillator_2_level | 0.1144 |
| oscillator_2_transpose | 0.0870 |
| envelope_1_attack | 0.0160 |
| envelope_1_decay | 0.0131 |
| envelope_2_attack | 0.0327 |
| envelope_2_decay | 0.0268 |
| filter_1_cutoff | 0.0397 |
| filter_1_resonance | 0.0707 |
| sample_level | 0.0624 |
| modulation_1_amount | 0.0361 |
| reverb_mix | 0.0453 |


### convnext-base-v2

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 36 | 0.0056 | 0.0657 |
| 1 | 36 | 0.0056 | 0.0642 |
| 2 | 37 | 0.0051 | 0.0652 |
| 3 | 37 | 0.0051 | 0.0645 |
| 4 | 35 | 0.0061 | 0.0640 |


### convnext-tiny-v2

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 34 | 0.0138 | 0.0676 |
| 1 | 35 | 0.0127 | 0.0666 |
| 2 | 36 | 0.0112 | 0.0665 |
| 3 | 36 | 0.0109 | 0.0661 |
| 4 | 37 | 0.0105 | 0.0657 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0666 | 0.0666 | 0.0668 | 0.0664 | 0.0661 | 0.0665 ± 0.0002 | 4 (0.0661) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1209 |
| oscillator_1_level | 0.0989 |
| oscillator_1_unison_voices | 0.1010 |
| oscillator_1_unison_detune | 0.0789 |
| oscillator_2_wave_frame | 0.1508 |
| oscillator_2_level | 0.1081 |
| oscillator_2_transpose | 0.0622 |
| envelope_1_attack | 0.0184 |
| envelope_1_decay | 0.0155 |
| envelope_2_attack | 0.0343 |
| envelope_2_decay | 0.0277 |
| filter_1_cutoff | 0.0378 |
| filter_1_resonance | 0.0665 |
| sample_level | 0.0583 |
| modulation_1_amount | 0.0359 |
| reverb_mix | 0.0426 |


### convnext-tiny-v2-nopitch

**Training (best eval MAE per split, with train loss at that epoch):**

| Split | Best epoch | Train loss @best | Best eval MAE |
|---|---|---|---|
| 0 | 36 | 0.0122 | 0.0690 |
| 1 | 33 | 0.0163 | 0.0676 |
| 2 | 38 | 0.0108 | 0.0678 |
| 3 | 33 | 0.0168 | 0.0675 |
| 4 | 34 | 0.0144 | 0.0678 |

**Testing (per split):**

| Split 0 | Split 1 | Split 2 | Split 3 | Split 4 | Avg ± std | Best split |
|---|---|---|---|---|---|---|
| 0.0684 | 0.0678 | 0.0682 | 0.0682 | 0.0679 | 0.0681 ± 0.0002 | 1 (0.0678) |

**Per-parameter MAE (best split):**

| Parameter | MAE |
|---|---|
| oscillator_1_wave_frame | 0.1205 |
| oscillator_1_level | 0.1005 |
| oscillator_1_unison_voices | 0.0987 |
| oscillator_1_unison_detune | 0.0802 |
| oscillator_2_wave_frame | 0.1442 |
| oscillator_2_level | 0.1100 |
| oscillator_2_transpose | 0.0776 |
| envelope_1_attack | 0.0190 |
| envelope_1_decay | 0.0153 |
| envelope_2_attack | 0.0363 |
| envelope_2_decay | 0.0273 |
| filter_1_cutoff | 0.0398 |
| filter_1_resonance | 0.0679 |
| sample_level | 0.0633 |
| modulation_1_amount | 0.0373 |
| reverb_mix | 0.0476 |


---
## V3  (multi-pitch ~168k + SpecAugment, single 80/10/10 split)

V3 has a single split (no 5-fold), and no per-split `test_results.txt`. Per-parameter and MSS come from the shared 100-sample eval set (see V2/V3 cross-testing).

**Training (single split, best eval MAE with train loss at that epoch):**

| Model | Best epoch | Train loss @best | Best eval MAE | Final-epoch train | Final-epoch eval |
|---|---|---|---|---|---|
| ast-small-v2 | 39 | 0.0318 | 0.0525 | 0.0317 | 0.0525 |
| ast-small-v2-nopitch | 39 | 0.0291 | 0.0563 | 0.0289 | 0.0563 |
| convnext-tiny-v2 | 31 | 0.0212 | 0.0534 | 0.0120 | 0.0537 |
| convnext-tiny-v2-nopitch | 33 | 0.0179 | 0.0554 | 0.0121 | 0.0558 |

---
## V2 / V3 cross-testing  (shared 100-sample eval set, identical for both)

**MAE + MSS (avg and per FFT scale), N=100:**

| Model | Ver | MAE | MSS avg | MSS 8192 | MSS 2048 | MSS 512 | MSS 128 |
|---|---|---|---|---|---|---|---|
| V2 ConvNeXt-tiny +pitch | v2 | 0.07391 | 1.40535 | 1.39954 | 1.34982 | 1.37866 | 1.49337 |
| V2 ConvNeXt-tiny | v2 | 0.07210 | 1.38284 | 1.39028 | 1.33692 | 1.34655 | 1.45760 |
| V2 AST-small +pitch | v2 | 0.07553 | 1.29154 | 1.27867 | 1.21803 | 1.26659 | 1.40288 |
| V2 AST-small | v2 | 0.07832 | 1.27321 | 1.27528 | 1.22659 | 1.23015 | 1.36083 |
| V3 ConvNeXt-tiny +pitch | v3 | 0.05296 | 1.22155 | 1.17333 | 1.14480 | 1.21324 | 1.35482 |
| V3 ConvNeXt-tiny | v3 | 0.05498 | 1.26357 | 1.23096 | 1.19669 | 1.25431 | 1.37232 |
| V3 AST-small +pitch | v3 | 0.05248 | 1.09566 | 1.04299 | 1.01850 | 1.09129 | 1.22987 |
| V3 AST-small | v3 | 0.05596 | 1.11903 | 1.07911 | 1.04749 | 1.10117 | 1.24834 |

**Per-parameter MAE on shared set (4 V2 models + 4 V3 models):**

| parameter | V2 ConvNeXt-tiny +pitch | V2 ConvNeXt-tiny | V2 AST-small +pitch | V2 AST-small | V3 ConvNeXt-tiny +pitch | V3 ConvNeXt-tiny | V3 AST-small +pitch | V3 AST-small |
|---|---|---|---|---|---|---|---|---|
| oscillator_1_wave_frame | 0.13547 | 0.12703 | 0.13347 | 0.13199 | 0.10111 | 0.10340 | 0.10147 | 0.10700 |
| oscillator_1_level | 0.10406 | 0.10143 | 0.12207 | 0.12023 | 0.08596 | 0.08914 | 0.08234 | 0.08767 |
| oscillator_1_unison_voices | 0.10584 | 0.11399 | 0.14040 | 0.13277 | 0.08063 | 0.08239 | 0.10221 | 0.11064 |
| oscillator_1_unison_detune | 0.07949 | 0.07451 | 0.08913 | 0.10545 | 0.05579 | 0.06036 | 0.06811 | 0.07297 |
| oscillator_2_wave_frame | 0.15160 | 0.14938 | 0.15589 | 0.16896 | 0.12333 | 0.12634 | 0.12051 | 0.12679 |
| oscillator_2_level | 0.11753 | 0.11838 | 0.12258 | 0.12236 | 0.08988 | 0.09396 | 0.08679 | 0.09213 |
| oscillator_2_transpose | 0.07594 | 0.07819 | 0.06725 | 0.08462 | 0.03133 | 0.03763 | 0.03081 | 0.03858 |
| envelope_1_attack | 0.02296 | 0.02108 | 0.02010 | 0.02045 | 0.01652 | 0.01806 | 0.01173 | 0.01410 |
| envelope_1_decay | 0.01891 | 0.02053 | 0.01836 | 0.01488 | 0.01159 | 0.01466 | 0.01032 | 0.01064 |
| envelope_2_attack | 0.04055 | 0.03862 | 0.03570 | 0.04086 | 0.02726 | 0.03063 | 0.02365 | 0.02457 |
| envelope_2_decay | 0.02955 | 0.02764 | 0.02674 | 0.02492 | 0.02470 | 0.02348 | 0.02128 | 0.02223 |
| filter_1_cutoff | 0.03984 | 0.03823 | 0.03942 | 0.03762 | 0.03225 | 0.03270 | 0.02935 | 0.03087 |
| filter_1_resonance | 0.08730 | 0.07887 | 0.07500 | 0.07386 | 0.05363 | 0.05372 | 0.04976 | 0.05226 |
| sample_level | 0.07795 | 0.07276 | 0.07543 | 0.07336 | 0.05177 | 0.05093 | 0.04572 | 0.04838 |
| modulation_1_amount | 0.03576 | 0.03764 | 0.03805 | 0.03964 | 0.03077 | 0.03222 | 0.02899 | 0.02992 |
| reverb_mix | 0.05983 | 0.05529 | 0.04886 | 0.06123 | 0.03087 | 0.03000 | 0.02661 | 0.02653 |
