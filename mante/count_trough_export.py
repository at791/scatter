"""Per-number 'trough and verdicts' sheets for the game page.  python count_trough_export.py -> count_trough.json
For each number n = 1..15: the trough is the spline through n's count states; the sheet spans the trough and one fixed
direction d_n, the mean direction in which the state travels from the trough to its end-of-scan verdict (rest blue and
can't tell trials, game timing, noise off), orthogonalised to the trough's mean tangent. On the sheet: log speed and flow
with scan-complete off and on, and the three readout regions (output > 0.5) baked into two PNGs.
JS projection of a live state x (300-D): arc position = arc length of the nearest curve point in the shared plane of
count_land.json (2-D, cheap); offset = x.d - c_j.d (one dot product per step)."""
import json, base64, io, numpy as np, count as c
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
Wt = dict(np.load('count_w_w_base.npz')); c.N = 300; c.NIN = 5; a = c.alpha_of(Wt)
land = json.load(open('count_land.json')); MU = np.array(land['mu']); PL2 = np.array(land['P'])
PRE, FIX, GAP, PUL, AMP, POST = 12, 12, 7, 3, 2.0, 50
def step(X, n, sc=0):
    u0 = np.zeros(5); u0[3] = n / 10; u0[4] = sc; return (1 - a) * X + a * (np.tanh(X) @ Wt['J'].T + u0 @ Wt['B'].T + Wt['c'])
zr = lambda Y: np.tanh(Y) @ Wt['W'].T + Wt['cz']
def trial(n, stones, rng):
    rows = []; blk = np.zeros((PRE, 5), np.float32); blk[:, 3] = n / 10; rows.append(blk)
    for k in stones:
        f, g = int(rng.integers(10, 16)), int(rng.integers(0, 9)); blk = np.zeros((f + g, 5), np.float32); blk[:PUL, 'bdh'.index(k)] = AMP; blk[:, 3] = n / 10; rows.append(blk)
    done = sum(len(r) for r in rows); blk = np.zeros((POST, 5), np.float32); blk[:, 3] = n / 10; blk[:, 4] = 1; rows.append(blk)
    U = np.concatenate(rows)[None]; z, (xs, _) = c.forward(Wt, U, None, noise=False, keep=True); return np.stack(xs[1:])[:, 0], z[0], done
def count_states(n):
    T = PRE + n * (FIX + GAP) + 60; U = np.zeros((1, T, 5), np.float32); t = PRE
    for j in range(n): U[0, t:t + PUL, 0] = AMP; t += FIX + GAP
    U[0, :, 3] = n / 10; _, (xs, _) = c.forward(Wt, U, None, noise=False, keep=True); X = np.stack(xs[1:])[:, 0]
    on = [PRE + j * (FIX + GAP) for j in range(n)]; return np.stack([X[o - 1] for o in on] + [X[on[-1] + FIX + GAP - 1]])
def catmull(Pts, m):
    Pp = np.vstack([Pts[0] - (Pts[1] - Pts[0]), Pts, Pts[-1] + (Pts[-1] - Pts[-2])]); out = []
    for i in range(1, len(Pp) - 2):
        p0, p1, p2, p3 = Pp[i - 1], Pp[i], Pp[i + 1], Pp[i + 2]
        for t in np.linspace(0, 1, m, endpoint=False): out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(Pp[-2]); return np.array(out)
SEG, NX, NY, PXW, PXH = 8, 70, 50, 300, 220
COLS = ['#2c2c2a', '#378add', '#ba7517']; out = dict(nums={})
rng = np.random.default_rng(3)
for n in range(1, 16):
    C = count_states(n); cv = catmull(C, SEG); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cv, axis=0), axis=1))]
    tan = np.gradient(cv, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True); tm = tan.mean(0); tm /= np.linalg.norm(tm)
    p2 = (cv - MU) @ PL2.T
    ends = []
    for stones in ('b' * (n - 1) + 'hd', 'b' * max(n - 2, 0) + 'hhhd'):                      # rest blue case, can't tell case
        for rep in range(3):
            X, z, done = trial(n, list(np.random.default_rng(rep + 10 * n).permutation(list(stones))), rng)
            hit = np.where((z[done:] > 0.5).any(1))[0]; te = done + (hit[0] + 10 if len(hit) else POST - 1)
            j = int(np.argmin(np.linalg.norm(((X[done] - MU) @ PL2.T) - p2, axis=1))); dv = X[min(te, len(X) - 1)] - cv[j]; ends.append(dv / np.linalg.norm(dv))
    d = np.mean(ends, 0); d -= (d @ tm) * tm; d /= np.linalg.norm(d)
    proj = [e @ d for e in ends]
    lo, hi = -3.0, 14.0
    xs_ = np.linspace(0, s[-1], NX); ys_ = np.linspace(lo, hi, NY)
    ci = np.clip(np.searchsorted(s, xs_), 0, len(s) - 1)
    G = (cv[ci][:, None, :] + ys_[None, :, None] * d[None, None, :]).reshape(-1, 300)
    Z = zr(G).reshape(NX, NY, 3); rec = dict(d=np.round(d, 5).tolist(), cx=np.round(p2[:, 0], 3).tolist(), cy=np.round(p2[:, 1], 3).tolist(), s=np.round(s, 3).tolist(),
                                            cd=np.round(cv @ d, 4).tolist(), ext=[0, float(s[-1]), lo, hi], ticks=np.round(s[::SEG], 3).tolist(), share=np.round((Z > 0.5).mean((0, 1)), 3).tolist())
    for sc in (0, 1):
        F = step(G, n, sc) - G; q = np.log10((F ** 2).sum(1) + 1e-12).reshape(NX, NY)
        fig = plt.figure(figsize=(PXW / 100, PXH / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.imshow(q.T, extent=[0, s[-1], lo, hi], origin='lower', aspect='auto', cmap='viridis', vmin=-1.8, vmax=0.9)
        for jj in range(3):
            if Z[:, :, jj].max() > 0.5:
                ax.contourf(xs_, ys_, Z[:, :, jj].T, levels=[0.5, 99], colors=[COLS[jj]], alpha=0.35); ax.contour(xs_, ys_, Z[:, :, jj].T, levels=[0.5], colors=[COLS[jj] if jj else '#ffffff'], linewidths=2)
        ax.set_xlim(0, s[-1]); ax.set_ylim(lo, hi); buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
        rec[f'png{sc}'] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        fa = np.linspace(0, s[-1], 14)[1:-1]; fb = np.linspace(lo, hi, 11)[1:-1]; fl = []
        for xv in fa:
            jc = int(np.clip(np.searchsorted(s, xv), 0, len(s) - 1))
            for yv in fb:
                g_ = cv[jc] + yv * d; Fv = step(g_[None], n, sc)[0] - g_; fl.append([round(float(xv), 3), round(float(yv), 3), round(float(Fv @ tan[jc]), 4), round(float(Fv @ d), 4)])
        rec[f'flow{sc}'] = fl
    out['nums'][str(n)] = rec; print(f'number {n}: verdict-direction spread {np.std(proj):.2f}, sheet share above 0.5 (rest dark / rest blue / can\'t tell) {rec["share"]}')
json.dump(out, open('count_trough.json', 'w'), separators=(',', ':')); print('KB', len(json.dumps(out)) // 1024)
