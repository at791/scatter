"""Speed landscapes for the artifact's live panel.  python count_landscape_export.py count_w_w_base.npz count_land.json
One shared plane: PCA of the net's states before each stone for every number cue n = 1..15 with k = 0..n blues seen (noise off,
pulse inputs). For each n, q(s) = |F(s) - s|^2 of the autonomous map (number n on, no stone) over a 90 x 90 grid of the plane,
log10, drawn with one colour scale for all n as a small PNG (base64). The JSON carries the plane (mean, two axes), the grid
extent, the colour range, and per number the PNG and the projected count states, so the page can project the live state."""
import sys, json, base64, io, numpy as np, count as c, count_var as v
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
Wt = dict(np.load(sys.argv[1])); N = Wt['J'].shape[0]; c.N = N; c.NIN = Wt['B'].shape[1]; c.MAXSIZE = 21; a = c.alpha_of(Wt)
c.make = v.make_gen(pulse=True); v.PAUSE_MAX = 0

def states_for(n):
    U, Y, M, meta = c.make([(n, ['b'] * n + ['h'] * 2)], np.random.default_rng(5 + n), interrupt=False)
    idx = np.array(meta[0]['idx']); onsets = [int(np.where(idx == k)[0][0]) for k in range(n + 1)]   # before stone k: k blues seen
    _, (xs, _) = c.forward(Wt, U, None, noise=False, keep=True)
    return np.stack([xs[t][0] for t in onsets])

def step(S, n):
    u0 = np.zeros(c.NIN); u0[3] = n / 10
    return (1 - a) * S + a * (np.tanh(S) @ Wt['J'].T + u0 @ Wt['B'].T + Wt['c'])

allS = {n: states_for(n) for n in range(1, 16)}
S = np.concatenate([allS[n] for n in allS]); mu = S.mean(0); _, sv, Vt = np.linalg.svd(S - mu, full_matrices=False); P = Vt[:2]
var2 = float((sv[:2] ** 2).sum() / (sv ** 2).sum()); proj = (S - mu) @ P.T
lo, hi = proj.min(0), proj.max(0); pad = 0.25 * (hi - lo); lo, hi = lo - pad, hi + pad
G = 90; ga = np.linspace(lo[0], hi[0], G); gb = np.linspace(lo[1], hi[1], G); A, B = np.meshgrid(ga, gb)
grid = mu + A.reshape(-1, 1) * P[0] + B.reshape(-1, 1) * P[1]
Q = {n: np.log10(((step(grid, n) - grid) ** 2).sum(1) + 1e-12).reshape(G, G) for n in allS}
vmin = float(min(q.min() for q in Q.values())); vmax = float(max(q.max() for q in Q.values()))
out = dict(N=N, var2=round(var2, 3), mu=np.round(mu, 4).tolist(), P=np.round(P, 5).tolist(), ext=[float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])], vmin=round(vmin, 2), vmax=round(vmax, 2), nums={})
F = 16; fa = np.linspace(lo[0], hi[0], F + 2)[1:-1]; fb = np.linspace(lo[1], hi[1], F + 2)[1:-1]; FA, FB = np.meshgrid(fa, fb)   # flow field: 16 x 16 arrows
fgrid = mu + FA.reshape(-1, 1) * P[0] + FB.reshape(-1, 1) * P[1]
flow = {n: ((step(fgrid, n) - fgrid) @ P.T) for n in allS}          # one-step displacement projected onto the plane (per 20 ms)
for n in allS:
    buf = io.BytesIO(); plt.imsave(buf, Q[n], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', format='png'); buf.seek(0)
    pts = ((allS[n] - mu) @ P.T).round(3).tolist()
    arrows = np.c_[FA.ravel(), FB.ravel(), flow[n]].round(4).tolist()
    out['nums'][str(n)] = dict(png='data:image/png;base64,' + base64.b64encode(buf.read()).decode(), pts=pts, flow=arrows)
json.dump(out, open(sys.argv[2], 'w'), separators=(',', ':'))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, n in zip(axes, (4, 8, 15)):
    ax.imshow(Q[n], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', extent=out['ext'], aspect='auto')
    fl = flow[n]; sp = np.linalg.norm(fl, axis=1); ax.quiver(FA, FB, fl[:, 0] / (sp + 1e-9), fl[:, 1] / (sp + 1e-9), sp, cmap='Greys_r', scale=28, width=0.004, headwidth=4, alpha=0.9)
    pts = np.array(out['nums'][str(n)]['pts']); ax.scatter(pts[:, 0], pts[:, 1], c='orangered', s=30, zorder=3, edgecolor='k', lw=0.4)
    for k, (x, y) in enumerate(pts): ax.annotate(str(k), (x, y), textcoords='offset points', xytext=(4, 3), color='white', fontsize=8)
    ax.set_title(f'number {n}: flow field (arrow direction) over log10 speed; dots = state after k blues', fontsize=9); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.tight_layout(); plt.savefig('count_flow.png', dpi=110)
print('plane var', round(var2, 3), 'extent', [round(x, 2) for x in out['ext']], 'log10 q range', round(vmin, 2), round(vmax, 2), 'json KB', round(len(json.dumps(out)) / 1024))
for n in (2, 8, 15): print('number', n, 'count-state speed log10', np.round(np.log10(((step(allS[n], n) - allS[n]) ** 2).sum(1)), 2).tolist())
