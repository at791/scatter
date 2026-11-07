"""Kinetic-energy landscape (Sussillo & Barak 2013) of a counting net's autonomous dynamics.  python count_landscape.py
Measured, not imposed: the count states are the net's own states before each stone of one scripted circle (number 15, 12 blues,
noise off); the plane is the first two PCs of those 13 states; q(s) = |F(s) - s|^2 is the one-step speed of the autonomous map
F with the number input held on and no stone (low q = slow or fixed). Also: q along the straight chain through the count states,
and 10 s autonomous drift from each state (noise off; and spread with standard noise). Rate net w_base vs the LSTM control.
Writes count_landscape.png and prints the numbers."""
import numpy as np, torch, count as c, count_var as v, count_torch as ct
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
dev = ct.dev; c.MAXSIZE = 21; HOLD = 500                                   # 10 s of autonomous dynamics

def trial():
    c.make = v.make_gen(pulse=True); c.NIN = 5; v.PAUSE_MAX = 0
    U, Y, M, meta = c.make([(15, ['b'] * 12 + ['h'] * 3)], np.random.default_rng(5), interrupt=False)
    idx = np.array(meta[0]['idx']); onsets = [int(np.where(idx == k)[0][0]) for k in range(13)]   # step at which stone k is first fixated
    u0 = np.zeros(5, np.float32); u0[3] = 1.5                             # number 15 on, nothing else (count.make: U[:, :, 3] = num / 10)
    return U, onsets, u0

class Rate:
    name = 'rate net w_base (300 tanh units)'; mem = slice(None)             # the whole state is the memory
    def __init__(self, path):
        self.Wt = dict(np.load(path)); self.a = c.alpha_of(self.Wt)
    def states(self, U, onsets):
        _, (xs, _) = c.forward(self.Wt, U, None, noise=False, keep=True)   # xs[t] = state after step t-1
        return np.stack([xs[t] [0] for t in onsets])
    def step(self, S, u0, noise=0.0, rng=None):
        W = self.Wt; drive = np.tanh(S) @ W['J'].T + u0 @ W['B'].T + W['c']
        S2 = (1 - self.a) * S + self.a * drive
        if noise: S2 = S2 + noise * np.sqrt(2 / c.ALPHA) * self.a * rng.standard_normal(S.shape)
        return S2

class Lstm:
    name = 'LSTM control (100 cells)'
    def __init__(self, path):
        import count_lstm as L; st = torch.load(path); L.cell.load_state_dict(st['cell']); self.cell = L.cell; self.H = L.H
        self.mem = slice(self.H, None)                                      # the cell state is the memory; h is its gated view
    def _run(self, U):
        h = torch.zeros(1, self.H, device=dev); cs = torch.zeros(1, self.H, device=dev); out = [torch.cat([h, cs], 1)]
        with torch.no_grad():
            for t in range(U.shape[1]):
                h, cs = self.cell(torch.tensor(U[:, t], device=dev), (h, cs)); out.append(torch.cat([h, cs], 1))
        return [o.cpu().numpy() for o in out]
    def states(self, U, onsets):
        xs = self._run(U); return np.stack([xs[t][0] for t in onsets])
    def step(self, S, u0, noise=0.0, rng=None):
        S = torch.tensor(S, dtype=torch.float32, device=dev); h, cs = S[:, :self.H], S[:, self.H:]
        with torch.no_grad(): h2, c2 = self.cell(torch.tensor(u0, device=dev).expand(S.shape[0], -1), (h, cs))
        S2 = torch.cat([h2, c2], 1).cpu().numpy()
        if noise: S2[:, :self.H] += noise * np.sqrt(2 / c.ALPHA) * c.ALPHA * rng.standard_normal((S.shape[0], self.H))
        return S2

def analyse(model, U, onsets, u0):
    S = model.states(U, onsets); mu = S.mean(0); _, sv, Vt = np.linalg.svd(S - mu, full_matrices=False)
    P = Vt[:2]; proj = (S - mu) @ P.T; var2 = (sv[:2] ** 2).sum() / (sv ** 2).sum()
    q = lambda X: ((model.step(X, u0) - X) ** 2).sum(1)
    lo, hi = proj.min(0), proj.max(0); pad = 0.35 * (hi - lo + 1e-9)
    ga = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 90); gb = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 90)
    A, Bg = np.meshgrid(ga, gb); G = mu + A.reshape(-1, 1) * P[0] + Bg.reshape(-1, 1) * P[1]
    Q = np.log10(q(G) + 1e-12).reshape(A.shape)
    # speed along the straight chain through the count states, 40 points per segment
    chain = np.concatenate([S[k] + (S[k + 1] - S[k]) * np.linspace(0, 1, 40, endpoint=False)[:, None] for k in range(12)] + [S[12:13]])
    qchain = q(chain); qstates = q(S)
    # 10 s autonomous drift from each count state: noise off (one path), standard noise (20 paths)
    X = S.copy(); path = [X.copy()]
    for _ in range(HOLD): X = model.step(X, u0); path.append(X.copy())
    drift = np.linalg.norm(path[-1] - S, axis=1); spacing = np.linalg.norm(S[1:] - S[:-1], axis=1)
    mem = model.mem; drift1 = np.linalg.norm(path[50] - S, axis=1)       # 1 s, inside the trained range of gap lengths
    drift_mem = np.linalg.norm((path[-1] - S)[:, mem], axis=1); drift1_mem = np.linalg.norm((path[50] - S)[:, mem], axis=1)
    spacing_mem = np.linalg.norm((S[1:] - S[:-1])[:, mem], axis=1)
    spacing_1s = np.linalg.norm((path[50][1:] - path[50][:-1])[:, mem], axis=1)      # do adjacent counts stay apart while the whole chain drifts?
    spacing_10s = np.linalg.norm((path[-1][1:] - path[-1][:-1])[:, mem], axis=1)
    print(model.name, '\n  memory spacing between adjacent counts: at the state', np.round(spacing_mem, 2), '\n    after 1 s', np.round(spacing_1s, 2), '\n    after 10 s', np.round(spacing_10s, 2))
    rng = np.random.default_rng(0); Xn = np.repeat(S, 20, 0)
    for _ in range(HOLD): Xn = model.step(Xn, u0, 0.05, rng)
    Xn = Xn.reshape(13, 20, -1); spread = np.linalg.norm(Xn - Xn.mean(1, keepdims=True), axis=2).mean(1); ndrift = np.linalg.norm(Xn.mean(1) - S, axis=1)
    traj = [(p - mu) @ P.T for p in path]                                  # noise-off paths in the plane
    return dict(S=S, proj=proj, var2=var2, ga=ga, gb=gb, Q=Q, qchain=qchain, qstates=qstates, drift=drift, spacing=spacing, spread=spread, ndrift=ndrift, traj=np.stack(traj, 1),
                drift1=drift1, drift_mem=drift_mem, drift1_mem=drift1_mem, spacing_mem=spacing_mem)

U, onsets, u0 = trial()
models = [Rate('count_w_w_base.npz'), Lstm('count_w_lstm.pt')]
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
for row, m in enumerate(models):
    r = analyse(m, U, onsets, u0)
    ax = axes[row, 0]; im = ax.pcolormesh(r['ga'], r['gb'], r['Q'], cmap='viridis', shading='auto'); plt.colorbar(im, ax=ax, label='log10 speed q (low = slow)')
    for k in range(13): ax.plot(r['traj'][k, :, 0], r['traj'][k, :, 1], color='white', lw=0.8, alpha=0.8)
    ax.scatter(r['proj'][:, 0], r['proj'][:, 1], c='orangered', s=36, zorder=3, edgecolor='k', lw=0.4)
    for k in range(13): ax.annotate(str(k), r['proj'][k], textcoords='offset points', xytext=(5, 4), color='white', fontsize=9)
    ax.set_title(f"{m.name}: speed on the PC1-PC2 plane of the 13 count states ({100 * r['var2']:.0f}% var)\n"
                 'dots = state after k blues (number 15); white = 10 s autonomous path, noise off', fontsize=10)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_xlim(r['ga'][0], r['ga'][-1]); ax.set_ylim(r['gb'][0], r['gb'][-1])
    ax = axes[row, 1]; xs = np.arange(len(r['qchain'])) / 40
    ax.semilogy(xs, r['qchain'] + 1e-12, color='tab:blue', lw=1.2, label='speed along the straight chain between count states')
    ax.semilogy(np.arange(13), r['qstates'] + 1e-12, 'o', color='orangered', label='speed at the count states')
    ax.set_xlabel('position along the chain (blues seen)'); ax.set_ylabel('speed q'); ax.legend(fontsize=8, loc='upper right')
    ax2 = ax.twinx(); ax2.bar(np.arange(13) - 0.27, r['drift1_mem'], 0.27, color='gray', alpha=0.7, label='memory drift in 1 s, noise off')
    ax2.bar(np.arange(13), r['drift_mem'], 0.27, color='gray', alpha=0.35, label='memory drift in 10 s, noise off')
    ax2.bar(np.arange(13) + 0.27, r['spread'], 0.27, color='tab:green', alpha=0.5, label='spread after 10 s, standard noise'); ax2.set_ylabel('distance (state units)')
    ax2.legend(fontsize=8, loc='center right')
    ax.set_title(f"{m.name}: barriers between counts? (dips at integers = wells, flat = line)\n"
                 f"memory spacing between adjacent counts {r['spacing_mem'].min():.2f}-{r['spacing_mem'].max():.2f}", fontsize=10)
    print(m.name, '\n  speed at count states', np.round(r['qstates'], 5), '\n  max speed between counts', np.round([r['qchain'][k * 40:(k + 1) * 40].max() for k in range(12)], 5),
          '\n  memory drift in 1 s, noise off', np.round(r['drift1_mem'], 2), '\n  memory drift in 10 s, noise off', np.round(r['drift_mem'], 2),
          '\n  whole-state drift in 1 s / 10 s', np.round(r['drift1'], 2), np.round(r['drift'], 2),
          '\n  10 s spread, standard noise', np.round(r['spread'], 2), '\n  memory spacing', np.round(r['spacing_mem'], 2), '\n  PC1+PC2 variance', round(r['var2'], 3))
plt.tight_layout(); plt.savefig('count_landscape.png', dpi=110); print('saved count_landscape.png')
