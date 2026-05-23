import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

W = 24   # window size matching dynamic_window in config
N = 200

np.random.seed(0)
t = np.arange(N)

# --- Archetypal signals ---

# High mean: sustained elevated signal
s_mean = 8 + 0.1 * np.random.randn(N)

# High std: large-amplitude oscillations fitting ~2 full cycles within the window
s_std = 1 * np.sin(2 * np.pi * t / (W / 2)) + 1 * np.random.randn(N)

# High gradient mean: monotonic ramp → gradient is large and consistently positive
s_grad_mean = np.cumsum(0.4 + 0.05 * np.random.randn(N))

# High gradient std: piecewise ramps with alternating steep/flat slopes →
# gradient flips between large and near-zero values, yielding high std(gradient)
seg_len = W 
n_segs = int(np.ceil(N / seg_len))
slopes = np.tile([3.0, 0.05, -3.0, 0.05], int(np.ceil(n_segs / 4)))[:n_segs]
grad_pattern = np.repeat(slopes, seg_len)[:N]
s_grad_std = np.cumsum(grad_pattern) + 0.2 * np.random.randn(N)

archetypes = [
    ('High window mean',      s_mean,      'rolling mean',          lambda s: pd.Series(s).rolling(W, center=True).mean(), [0,10]),
    ('High window std',       s_std,       'rolling std',           lambda s: pd.Series(s).rolling(W, center=True).std(),[-2,2]),
    ('High gradient mean',    s_grad_mean, 'rolling gradient mean', lambda s: pd.Series(s).diff().rolling(W, center=True).mean(),[0,100]),
    ('High gradient std',     s_grad_std,  'rolling gradient std',  lambda s: pd.Series(s).diff().rolling(W, center=True).std(),[0,90]),
]

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

for i, (title, sig, feat_label, feat_fn, ylim) in enumerate(archetypes):
    feature = feat_fn(sig)

    ax = axes[i]
    ax.plot(t, sig, color='k', linewidth=0.8, label='signal')
    ax.set_ylabel('signal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_ylim(ylim)

    ax2 = ax.twinx()
    ax2.plot(t, feature, color='steelblue', linewidth=1.2, alpha=0.85, label=feat_label)
    ax2.set_ylabel(feat_label, fontsize=9, color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=8)
    ax2.spines['right'].set_edgecolor('steelblue')

    if i == 0:
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=8)

axes[-1].set_xlabel('time step')
fig.suptitle(f'Dynamic feature archetypes  (window W={W})', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()

out_dir = Path('figures')
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir / 'dynamic_feature_archetypes.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
# plt.close(fig)
print(f"Saved {out_path}")
