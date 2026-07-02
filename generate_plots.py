"""
generate_plots.py
Generates all five result figures for the capstone report.
Saves PDFs to figs/ for direct \includegraphics use in LaTeX.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── output directory ──────────────────────────────────────────────────────────
FIGS_DIR = r"D:\downloads\Template_ISS499 (2)\Template ISS499\figs"
os.makedirs(FIGS_DIR, exist_ok=True)

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          10,
    'axes.titlesize':     10.5,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    8.5,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'axes.grid.axis':     'y',
    'grid.alpha':         0.35,
    'grid.linestyle':     '--',
    'grid.color':         '#bbbbbb',
})

# ── colour constants ──────────────────────────────────────────────────────────
GRAY       = '#9e9e9e'
BLUE_LT    = '#9ecae1'
BLUE_DK    = '#2171b5'
ORANGE_LT  = '#fdae6b'
ORANGE_DK  = '#d95f02'

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — V1 Architecture × Pitch-Conditioning Comparison
# ══════════════════════════════════════════════════════════════════════════════
v1 = {
    'CNN':            {'No Pitch': (0.0850, 0.0014), 'Basic MLP': (0.0828, 0.0007), 'Sinusoidal': (0.0829, 0.0008)},
    'AST-small':      {'No Pitch': (0.0793, 0.0006), 'Basic MLP': (0.0781, 0.0005), 'Sinusoidal': (0.0776, 0.0007)},
    'ConvNeXt-tiny':  {'No Pitch': (0.0767, 0.0004), 'Basic MLP': (0.0765, 0.0007), 'Sinusoidal': (0.0758, 0.0005)},
}
backbones   = ['CNN', 'AST-small', 'ConvNeXt-tiny']
modes       = ['No Pitch', 'Basic MLP', 'Sinusoidal']
m_colors    = [GRAY, BLUE_LT, BLUE_DK]
offsets     = [-0.27, 0.0, 0.27]

fig, ax = plt.subplots(figsize=(7.5, 4.0))
x = np.arange(len(backbones))

for mode, color, off in zip(modes, m_colors, offsets):
    means = [v1[b][mode][0] for b in backbones]
    stds  = [v1[b][mode][1] for b in backbones]
    ax.bar(x + off, means, 0.24, label=mode, color=color,
           yerr=stds, capsize=3,
           error_kw={'elinewidth': 1.0, 'ecolor': '#555555', 'capthick': 1.0},
           edgecolor='white', linewidth=0.5)

ax.axhline(0.0793, color='#e41a1c', linewidth=1.2, linestyle=':',
           label='Bruford et al. reference (AST no-pitch)')
ax.set_xticks(x)
ax.set_xticklabels(backbones)
ax.set_ylabel('Test MAE  (↓ better)')
ax.set_ylim(0.068, 0.098)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax.set_title('V1: Architecture × Pitch Conditioning — 5-Fold Test MAE')
ax.legend(framealpha=0.95, loc='upper right', ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig_v1_architecture.pdf'), bbox_inches='tight')
plt.close(fig)
print("OK fig_v1_architecture.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Training Convergence: V2 Overfitting vs V3 Generalisation
# ══════════════════════════════════════════════════════════════════════════════
ep     = [1,  5, 10, 15, 20, 25, 30, 35, 40]

v2_tr  = [0.2295, 0.1462, 0.1155, 0.0909, 0.0656, 0.0416, 0.0232, 0.0124, 0.0095]
v2_val = [0.1332, 0.0985, 0.0920, 0.0810, 0.0775, 0.0706, 0.0684, 0.0676, 0.0676]

v3_tr  = [0.1955, 0.1280, 0.1139, 0.1001, 0.0844, 0.0672, 0.0489, 0.0360, 0.0317]
v3_val = [0.1091, 0.0797, 0.0738, 0.0663, 0.0604, 0.0573, 0.0543, 0.0529, 0.0525]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)

panels = [
    (axes[0], v2_tr, v2_val, 'V2 — ConvNeXt-tiny + FiLM  (split 0)'),
    (axes[1], v3_tr, v3_val, 'V3 — AST-small + FiLM'),
]
for ax, tr, val, title in panels:
    ax.plot(ep, tr,  'o-',  color=BLUE_DK,   lw=1.8, ms=4, label='Train loss (SmoothL1)')
    ax.plot(ep, val, 's--', color=ORANGE_DK, lw=1.8, ms=4, label='Val MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SmoothL1 loss  /  MAE')
    ax.set_title(title)
    ax.set_xlim(0, 42)
    ax.legend(framealpha=0.95)

# shade overfitting gap on V2 panel
axes[0].fill_between(ep, v2_tr, v2_val,
                     where=[t < v for t, v in zip(v2_tr, v2_val)],
                     alpha=0.15, color='red', label='Train-loss / val-MAE gap')
axes[0].legend(framealpha=0.95)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle('Training Convergence: V2 Overfitting vs V3 Generalisation', fontsize=10.5)
fig.savefig(os.path.join(FIGS_DIR, 'fig_convergence.pdf'), bbox_inches='tight')
plt.close(fig)
print("OK fig_convergence.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — FiLM Conditioning Gap: V2 vs V3
# ══════════════════════════════════════════════════════════════════════════════
groups = ['V2\nConvNeXt', 'V2\nAST-small', 'V3\nConvNeXt', 'V3\nAST-small']
film_vals    = [0.0665,  0.0713,  0.0530, 0.0525]
nopitch_vals = [0.0681,  0.0727,  0.0550, 0.0560]
gaps         = [n - f for f, n in zip(film_vals, nopitch_vals)]

film_cols    = [BLUE_LT,  ORANGE_LT,  BLUE_DK,   ORANGE_DK]
nopitch_cols = ['#d9d9d9', '#fdd0a2', '#9ecae1', '#fdae6b']

x = np.arange(len(groups))
w = 0.33

fig, ax = plt.subplots(figsize=(7.5, 4.0))
ax.bar(x - w/2, film_vals,    w * 0.9, color=film_cols,    edgecolor='white',
       label='FiLM (with pitch)')
ax.bar(x + w/2, nopitch_vals, w * 0.9, color=nopitch_cols, edgecolor='white',
       hatch='////', alpha=0.85, label='No-pitch')

for i, (xp, gap) in enumerate(zip(x, gaps)):
    ypos = max(film_vals[i], nopitch_vals[i]) + 0.0005
    ax.annotate(f'Δ = {gap:.4f}', xy=(xp, ypos),
                ha='center', va='bottom', fontsize=8.5, color='#333333')

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=9.5)
ax.set_ylabel('Test MAE  (↓ better)')
ax.set_title('FiLM Conditioning Gap: V2 vs V3')
ax.set_ylim(0.046, 0.084)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

legend_elems = [
    Patch(facecolor=BLUE_DK,  label='ConvNeXt FiLM'),
    Patch(facecolor=BLUE_LT,  hatch='////', edgecolor='gray', label='ConvNeXt No-pitch'),
    Patch(facecolor=ORANGE_DK, label='AST FiLM'),
    Patch(facecolor=ORANGE_LT, hatch='////', edgecolor='gray', label='AST No-pitch'),
]
ax.legend(handles=legend_elems, ncol=2, framealpha=0.95, loc='upper right')

fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig_film_gap.pdf'), bbox_inches='tight')
plt.close(fig)
print("OK fig_film_gap.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Per-Parameter MAE: V1 → V2 → V3
# ══════════════════════════════════════════════════════════════════════════════
# Canonical grouped-head order: Oscillator | Envelope | Filter | Misc
PARAM_KEYS = [
    'oscillator_1_wave_frame', 'oscillator_1_level',
    'oscillator_1_unison_voices', 'oscillator_1_unison_detune',
    'oscillator_2_wave_frame', 'oscillator_2_level', 'oscillator_2_transpose',
    'envelope_1_attack', 'envelope_1_decay', 'envelope_2_attack', 'envelope_2_decay',
    'filter_1_cutoff', 'filter_1_resonance',
    'sample_level', 'modulation_1_amount', 'reverb_mix',
]
PARAM_LABELS = [
    'Osc1\nWave', 'Osc1\nLevel', 'Osc1\nUnison', 'Osc1\nDetune',
    'Osc2\nWave', 'Osc2\nLevel', 'Osc2\nTransp.',
    'Env1\nAtk', 'Env1\nDec', 'Env2\nAtk', 'Env2\nDec',
    'Filt\nCutoff', 'Filt\nReson.',
    'Sample\nLevel', 'Mod\nAmt', 'Reverb\nMix',
]

v1_pp = {
    'oscillator_1_wave_frame': 0.1362, 'oscillator_1_level': 0.1161,
    'oscillator_1_unison_voices': 0.1067, 'oscillator_1_unison_detune': 0.0869,
    'oscillator_2_wave_frame': 0.1610, 'oscillator_2_level': 0.1288,
    'oscillator_2_transpose': 0.0902,
    'envelope_1_attack': 0.0245, 'envelope_1_decay': 0.0191,
    'envelope_2_attack': 0.0367, 'envelope_2_decay': 0.0269,
    'filter_1_cutoff': 0.0415, 'filter_1_resonance': 0.0679,
    'sample_level': 0.0809, 'modulation_1_amount': 0.0346, 'reverb_mix': 0.0461,
}
v2_pp = {
    'oscillator_1_wave_frame': 0.1209, 'oscillator_1_level': 0.0989,
    'oscillator_1_unison_voices': 0.1010, 'oscillator_1_unison_detune': 0.0789,
    'oscillator_2_wave_frame': 0.1508, 'oscillator_2_level': 0.1081,
    'oscillator_2_transpose': 0.0622,
    'envelope_1_attack': 0.0184, 'envelope_1_decay': 0.0155,
    'envelope_2_attack': 0.0343, 'envelope_2_decay': 0.0277,
    'filter_1_cutoff': 0.0378, 'filter_1_resonance': 0.0665,
    'sample_level': 0.0583, 'modulation_1_amount': 0.0359, 'reverb_mix': 0.0426,
}
v3_pp = {
    'oscillator_1_wave_frame': 0.10147, 'oscillator_1_level': 0.08234,
    'oscillator_1_unison_voices': 0.10221, 'oscillator_1_unison_detune': 0.06811,
    'oscillator_2_wave_frame': 0.12051, 'oscillator_2_level': 0.08679,
    'oscillator_2_transpose': 0.03081,
    'envelope_1_attack': 0.01173, 'envelope_1_decay': 0.01032,
    'envelope_2_attack': 0.02365, 'envelope_2_decay': 0.02128,
    'filter_1_cutoff': 0.02935, 'filter_1_resonance': 0.04976,
    'sample_level': 0.04572, 'modulation_1_amount': 0.02899, 'reverb_mix': 0.02661,
}

v1_vals = [v1_pp[k] for k in PARAM_KEYS]
v2_vals = [v2_pp[k] for k in PARAM_KEYS]
v3_vals = [v3_pp[k] for k in PARAM_KEYS]

n = len(PARAM_KEYS)
x = np.arange(n)
w = 0.26

fig, ax = plt.subplots(figsize=(14.5, 4.5))

ax.bar(x - w, v1_vals, w * 0.9, label='V1 best  (ConvNeXt + Sinusoidal)',
       color='#bdbdbd', edgecolor='white')
ax.bar(x,     v2_vals, w * 0.9, label='V2 best  (ConvNeXt + FiLM)',
       color=BLUE_DK, edgecolor='white')
ax.bar(x + w, v3_vals, w * 0.9, label='V3 best  (AST + FiLM)',
       color=ORANGE_DK, edgecolor='white')

# Group background shading
group_shading = [
    (0,  6,  '#ddeeff', 'Oscillator'),
    (7,  10, '#e8f5e9', 'Envelope'),
    (11, 12, '#fce4ec', 'Filter'),
    (13, 15, '#fff3e0', 'Misc'),
]
for s, e, col, lbl in group_shading:
    ax.axvspan(s - 0.5, e + 0.5, alpha=0.22, color=col, zorder=0)
    ax.text((s + e) / 2, 0.175, lbl, ha='center', va='top',
            fontsize=8, color='#555555', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(PARAM_LABELS, fontsize=7.8)
ax.set_ylabel('MAE  (↓ better)')
ax.set_ylim(0, 0.185)
ax.set_xlim(-0.6, n - 0.4)
ax.set_title('Per-Parameter MAE: V1 → V2 → V3 Best Models')
ax.legend(framealpha=0.95, loc='upper right', ncol=3)

fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig_per_param.pdf'), bbox_inches='tight')
plt.close(fig)
print("OK fig_per_param.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — MSS Breakdown Across FFT Scales
# ══════════════════════════════════════════════════════════════════════════════
fft_labels = ['128\n(energy)', '512\n(transient)', '2048\n(timbral)', '8192\n(harmonic)']

mss = {
    'V2 ConvNeXt + FiLM': ([1.49337, 1.37866, 1.34982, 1.39954], BLUE_LT,   '--', 'o', 1.6),
    'V2 AST + FiLM':      ([1.40288, 1.26659, 1.21803, 1.27867], ORANGE_LT, '--', 's', 1.6),
    'V3 ConvNeXt + FiLM': ([1.35482, 1.21324, 1.14480, 1.17333], BLUE_DK,   '-',  'o', 2.2),
    'V3 AST + FiLM':      ([1.22987, 1.09129, 1.01850, 1.04299], ORANGE_DK, '-',  's', 2.2),
}

x = np.arange(4)
fig, ax = plt.subplots(figsize=(7.0, 4.0))

for label, (vals, color, ls, marker, lw) in mss.items():
    ax.plot(x, vals, marker=marker, linestyle=ls, linewidth=lw,
            markersize=6.5, color=color, label=label)

ax.set_xticks(x)
ax.set_xticklabels(fft_labels)
ax.set_ylabel('MSS Loss  (↓ better)')
ax.set_xlabel('FFT Scale  (n_fft)')
ax.set_title('Multi-Scale Spectral Loss Across FFT Scales (N = 100, shared eval set)')
ax.set_ylim(0.95, 1.60)
ax.legend(framealpha=0.95, loc='upper left')

fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig_mss_scales.pdf'), bbox_inches='tight')
plt.close(fig)
print("OK fig_mss_scales.pdf")

print(f"\nAll figures saved to:  {FIGS_DIR}")
