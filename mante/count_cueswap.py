"""Is 'the count' a location in state space, or a position relative to the current number's trough end?
python count_cueswap.py count_w_w_base.npz   (noise off, pulse stones, 300-unit net; plane from count_land.json)
A: the count states of two numbers on the shared plane, with the equal-count and equal-need pairs marked and their 300-D distances.
B: for every count state (number n, k blues seen) the nearest count state of a *different* number: does it share k (blues seen)
   or n-k (blues still needed)?
C: cue swap mid-trial: k blues under number n, then the number input switches to m and blues keep coming. If the state stored
   'k blues seen' the net stops after m-k more; if it stored position relative to the trough end (n-k needed) it stops after n-k."""
import sys, json, numpy as np, count as c
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
Wt = dict(np.load(sys.argv[1] if len(sys.argv) > 1 else 'count_w_w_base.npz')); c.N = Wt['J'].shape[0]; c.NIN = 5
land = json.load(open('count_land.json')); mu = np.array(land['mu']); P = np.array(land['P'])
PRE, FIX, GAP, PL, AMP = 12, 12, 7, 3, 2.0                     # steps: number-only lead-in, fixation, gap; onset pulse length / amplitude

def build(n, k, m=None, extra=20):
    """Number n on from the start; blue stones as onset pulses; after stone k the number input becomes m (None = stays n).
    Returns U (1,T,5), onset steps of each stone, and the swap step."""
    T = PRE + (k + extra) * (FIX + GAP) + 30
    U = np.zeros((1, T, 5), np.float32); on = []; t = PRE
    for j in range(k + extra):
        U[0, t:t + PL, 0] = AMP; on.append(t); t += FIX + GAP
    swap = PRE + k * (FIX + GAP) - GAP                             # start of the gap after stone k (= PRE for k = 0)
    U[0, :, 3] = n / 10
    if m is not None: U[0, swap:, 3] = m / 10
    return U, on, swap

def run(U):
    z, (xs, _) = c.forward(Wt, U, None, noise=False, keep=True)
    return z[0], np.stack(xs[1:])[:, 0]                            # xs[t+1] = state after step t, aligned with z[t]

# ---- states before each stone: S[n][k] = state with number n on and k blues seen (k = 0..n) ----
S = {}
for n in range(1, 16):
    U, on, _ = build(n, n, None, extra=1); _, X = run(U)
    S[n] = np.stack([X[on[k] - 1] for k in range(n + 1)])          # state just before stone k+1 starts

# ---- B: nearest count state of another number ----
keys = [(n, k) for n in S for k in range(n + 1)]; A = np.stack([S[n][k] for n, k in keys])
same_k = same_need = 0; rows = []
for i, (n, k) in enumerate(keys):
    d = np.linalg.norm(A - A[i], axis=1); d[[j for j, (n2, _) in enumerate(keys) if n2 == n]] = np.inf
    j = int(np.argmin(d)); n2, k2 = keys[j]
    same_k += k2 == k; same_need += (n2 - k2) == (n - k); rows.append((n, k, n2, k2, d[j]))
print(f'B: {len(keys)} count states; nearest state of another number shares blues-seen k in {same_k}, shares need n-k in {same_need}')

# ---- C: cue swap ----
NUMS = [3, 5, 8, 11]; res = []
for n in NUMS:
    for m in NUMS:
        for k in range(1, n):
            U, on, swap = build(n, k, m); z, _ = run(U)
            hit = np.where(z[swap:, 0] > 0.5)[0]
            if len(hit) == 0: extra = np.nan
            else:
                t = swap + hit[0]; extra = sum(1 for o in on if o <= t) - k
            res.append((n, m, k, extra))
res = np.array(res, float)
ctrl = res[res[:, 0] == res[:, 1]]; print('C control (m = n): observed extra - (n-k):', np.unique(ctrl[:, 3] - (ctrl[:, 0] - ctrl[:, 2]), return_counts=True))
sw = res[res[:, 0] != res[:, 1]]; need = sw[:, 0] - sw[:, 2]; cnt = sw[:, 1] - sw[:, 2]; ok = ~np.isnan(sw[:, 3])
print(f'C swaps: {len(sw)} trials, {int((~ok).sum())} never answered; matches need-preserved (n-k): {int((sw[ok, 3] == need[ok]).sum())}; '
      f'matches count-preserved (m-k): {int((sw[ok, 3] == cnt[ok]).sum())}; neither: {int(((sw[ok, 3] != need[ok]) & (sw[ok, 3] != cnt[ok])).sum())}')
for r in sw: print('  n=%d k=%d -> m=%d : stopped after %s more blues (need-pres %d, count-pres %d)' % (r[0], r[2], r[1], 'never' if np.isnan(r[3]) else int(r[3]), r[0] - r[2], r[1] - r[2]))

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
na, nb = 5, 8; pa = (S[na] - mu) @ P.T; pb = (S[nb] - mu) @ P.T
ax[0].plot(pa[:, 0], pa[:, 1], '-o', color='tab:blue', label=f'number {na}: states after k blues'); ax[0].plot(pb[:, 0], pb[:, 1], '-o', color='tab:orange', label=f'number {nb}')
for k, (x, y) in enumerate(pa): ax[0].annotate(str(k), (x, y), textcoords='offset points', xytext=(-10, 4), color='tab:blue', fontsize=9)
for k, (x, y) in enumerate(pb): ax[0].annotate(str(k), (x, y), textcoords='offset points', xytext=(5, -10), color='tab:orange', fontsize=9)
ka = 2; d_cnt = np.linalg.norm(S[na][ka] - S[nb][ka]); d_need = np.linalg.norm(S[na][ka] - S[nb][nb - (na - ka)])
ax[0].plot([pa[ka, 0], pb[ka, 0]], [pa[ka, 1], pb[ka, 1]], '--', color='grey'); ax[0].plot([pa[ka, 0], pb[nb - na + ka, 0]], [pa[ka, 1], pb[nb - na + ka, 1]], '-', color='crimson')
ax[0].text(0.02, 0.04, f'number {na}, {ka} blues seen (need {na - ka})\n-- to number {nb}, {ka} seen (need {nb - ka}): 300-D distance {d_cnt:.1f}\n-- to number {nb}, {nb - na + ka} seen (need {na - ka}): distance {d_need:.1f}', transform=ax[0].transAxes, fontsize=8.5, va='bottom', bbox=dict(fc='white', ec='none', alpha=.8))
U, on, swap = build(na, ka, nb, extra=0); _, X = run(U); tr = (X[swap - 1:] - mu) @ P.T      # the state after the cue switches 5 -> 8, no further stones
ax[0].plot(tr[:, 0], tr[:, 1], '-', color='green', lw=2, alpha=.8); ax[0].annotate('', tr[-1], tr[-8], arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax[0].text(tr[-1, 0], tr[-1, 1], f'  cue switched {na}→{nb}\n  ({(len(tr) - 1) * 20} ms, no stones)', color='green', fontsize=8.5, va='center')
print(f'A: 300-D distance from ({na},{ka}) to ({nb},{ka}) = {d_cnt:.2f}, to ({nb},{nb - na + ka}) = {d_need:.2f}; after the switch, nearest {nb}-state is k = {int(np.argmin(np.linalg.norm(S[nb] - X[-1], axis=1)))}')
ax[0].set_title('A  two numbers on the shared counting plane', fontsize=10); ax[0].set_xlabel('PC1'); ax[0].set_ylabel('PC2'); ax[0].legend(fontsize=8, loc='upper right')
dk = np.array([r[3] - r[1] for r in rows]); dneed = np.array([(r[2] - r[3]) - (r[0] - r[1]) for r in rows]); b = np.arange(-8.5, 9.5)
ax[1].hist(dk, bins=b, alpha=.6, label="nearest state's k  −  my k"); ax[1].hist(dneed, bins=b, alpha=.6, label="nearest state's need  −  my need")
ax[1].set_title(f'B  nearest count state of another number ({len(keys)} states)\nsame k: {same_k}   same need: {same_need}', fontsize=10); ax[1].set_xlabel('difference'); ax[1].legend(fontsize=8)
feas = ok & (cnt >= 1)                                                     # trials where 'k blues already seen' still leaves blues to come under m
e_cnt = sw[feas, 3] - cnt[feas]; e_need = sw[feas, 3] - need[feas]
print(f'C ({int(feas.sum())} feasible swaps): observed − (m−k): exact {int((e_cnt == 0).sum())}, within ±1 {int((abs(e_cnt) <= 1).sum())};  observed − (n−k): exact {int((e_need == 0).sum())}, within ±1 {int((abs(e_need) <= 1).sum())}')
ax[2].hist(e_cnt, bins=b, alpha=.6, label='observed − (m−k): blues seen kept, new number applied'); ax[2].hist(e_need, bins=b, alpha=.6, label='observed − (n−k): blues still needed kept')
ax[2].set_xlabel('extra blues before "rest dark": observed − predicted'); ax[2].set_ylabel('trials')
ax[2].set_title(f'C  cue switched n→m after k blues, blues keep coming ({int(feas.sum())} trials)\n(+{int((~feas).sum())} trials with k ≥ m, already past the new end: answered {int(np.nanmin(sw[~feas, 3]))}–{int(np.nanmax(sw[~feas, 3]))} blues later)', fontsize=9.5); ax[2].legend(fontsize=8, loc='upper left')
plt.tight_layout(); plt.savefig('count_cueswap.png', dpi=110); print('saved count_cueswap.png')
