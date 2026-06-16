import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

AUDIO_PATH = r"C:\Users\Splinter\Documents\pfe\testdata\bass_note36.wav"
OUT_PATH   = r"D:\downloads\Template_ISS499 (2)\Template ISS499\images\spectrogram_comparison.png"

y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
print(f"Loaded: sr={sr} Hz, duration={len(y)/sr:.2f}s")

N_FFT  = 2048
HOP    = 512

# Linear spectrogram
D    = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Mel spectrogram
N_MELS = 128
M      = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP, n_mels=N_MELS,
                                         fmin=20, fmax=12000, power=2.0)
M_db   = librosa.power_to_db(M, ref=np.max)

fig = plt.figure(figsize=(13, 4.5))
gs  = gridspec.GridSpec(1, 2, wspace=0.38)

ax1 = fig.add_subplot(gs[0])
img1 = librosa.display.specshow(S_db, sr=sr, hop_length=HOP,
                                 x_axis='time', y_axis='hz',
                                 ax=ax1, cmap='magma')
ax1.set_title('(a) Spectrogram (linear frequency)', fontsize=11, pad=8)
ax1.set_xlabel('Time (s)', fontsize=10)
ax1.set_ylabel('Frequency (Hz)', fontsize=10)
cb1 = fig.colorbar(img1, ax=ax1, format='%+2.0f dB', pad=0.02)
cb1.ax.tick_params(labelsize=8)

ax2 = fig.add_subplot(gs[1])
img2 = librosa.display.specshow(M_db, sr=sr, hop_length=HOP,
                                 x_axis='time', y_axis='mel',
                                 fmin=20, fmax=12000,
                                 ax=ax2, cmap='magma')
ax2.set_title('(b) Mel Spectrogram (128 mel bins)', fontsize=11, pad=8)
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Frequency (mel)', fontsize=10)
cb2 = fig.colorbar(img2, ax=ax2, format='%+2.0f dB', pad=0.02)
cb2.ax.tick_params(labelsize=8)

fig.suptitle('Bass synthesizer note (MIDI 36) rendered by Vital',
             fontsize=10, style='italic', y=1.01)

plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PATH}")
plt.close()
