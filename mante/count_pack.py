"""Pack a counting-net checkpoint for the artifact's JS player.  python count_pack.py count_w_w_base.npz count_b64.txt
Float32 J, B, c, W, cz, then per-unit alpha (dt/tau; filled with the standard 0.2 if the checkpoint has none), base64."""
import sys, base64, numpy as np, count as c
Wt = dict(np.load(sys.argv[1])); N = Wt['J'].shape[0]
alpha = np.asarray(Wt['alpha'], float) * np.ones(N) if 'alpha' in Wt else np.full(N, c.ALPHA)
f = np.concatenate([Wt[k].astype(np.float32).ravel() for k in ('J', 'B', 'c', 'W', 'cz')] + [alpha.astype(np.float32)])
open(sys.argv[2], 'w').write(base64.b64encode(f.tobytes()).decode())
print('packed', sys.argv[1], 'N', N, 'NIN', Wt['B'].shape[1], 'floats', f.size, 'pulse', bool(Wt.get('pulse', False)), 'alpha range', round(float(alpha.min()), 4), round(float(alpha.max()), 4))
