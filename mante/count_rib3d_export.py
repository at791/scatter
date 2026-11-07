"""3-D ribbon data for the game page.  python count_rib3d_export.py -> count_rib3d.json
Axes: PC1-PC3 of the count states of all 15 numbers (mu3, P3 exported so the page projects the live state itself).
Per number: its count states in 3-D (the trough), and a ribbon along it (spline, rotation-minimising normal, half-width 2.5)
as a grid of 3-D points with per-cell log speed (that number on, scan-complete off) and a flag where rest dark > 0.5; plus a
default view direction (normal of the ribbon's best-fit plane, kept >= 20 degrees above edge-on)."""
import json, numpy as np, count as c
Wt = dict(np.load('count_w_w_base.npz')); c.N = 300; c.NIN = 5; a = c.alpha_of(Wt)
PRE, FIX, GAP, PUL, AMP = 12, 12, 7, 3, 2.0
def step(X, n):
    u0 = np.zeros(5); u0[3] = n / 10; return (1 - a) * X + a * (np.tanh(X) @ Wt['J'].T + u0 @ Wt['B'].T + Wt['c'])
zr = lambda Y: np.tanh(Y) @ Wt['W'].T + Wt['cz']
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
S = {n: count_states(n) for n in range(1, 16)}; A = np.concatenate(list(S.values())); mu = A.mean(0); _, sv, Vt = np.linalg.svd(A - mu, full_matrices=False); P = Vt[:3]
pr = lambda X: (X - mu) @ P.T
W, K, SEG = 2.5, 9, 6; off = np.linspace(-W, W, K)
out = dict(mu=np.round(mu, 4).tolist(), P=np.round(P, 5).tolist(), var=np.round(sv[:3] ** 2 / (sv ** 2).sum(), 3).tolist(), ext=[np.round(pr(A).min(0) - 2, 2).tolist(), np.round(pr(A).max(0) + 2, 2).tolist()], nums={})
for n in range(1, 16):
    cv = catmull(S[n], SEG); M = len(cv); tan = np.gradient(cv, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    n1 = np.zeros_like(cv); v = Vt[2] - (Vt[2] @ tan[0]) * tan[0]; n1[0] = v / np.linalg.norm(v)
    for i in range(1, M): v = n1[i - 1] - (n1[i - 1] @ tan[i]) * tan[i]; n1[i] = v / np.linalg.norm(v)
    R = (cv[:, None, :] + off[None, :, None] * n1[:, None, :]).reshape(-1, 300); F = step(R, n) - R
    q = np.log10((F ** 2).sum(1) + 1e-12).reshape(M, K); rd = (zr(R)[:, 0] > 0.5).reshape(M, K); p3 = pr(R).reshape(M, K, 3)
    vv = np.linalg.svd(p3.reshape(-1, 3) - p3.reshape(-1, 3).mean(0), full_matrices=False)[2][2]
    if vv[2] < 0: vv = -vv
    if np.degrees(np.arcsin(np.clip(vv[2], -1, 1))) < 20: vv = vv + np.array([0, 0, 0.45]); vv /= np.linalg.norm(vv)
    ft = (F.reshape(M, K, 300) * tan[:, None, :]).sum(-1); f1 = (F.reshape(M, K, 300) * n1[:, None, :]).sum(-1)       # flow arrows on the ribbon, as in the mockup
    dt3 = (tan @ P.T); d13 = (n1 @ P.T); ar = []
    for i in range(2, M, 4):
        for j in (1, 4, 7):
            vv3 = ft[i, j] * dt3[i] + f1[i, j] * d13[i]; nv = np.linalg.norm(vv3) + 1e-9; ar.append([*np.round(p3[i, j], 3).tolist(), *np.round(vv3 / nv, 3).tolist(), round(float(np.hypot(ft[i, j], f1[i, j])), 4)])
    out['nums'][str(n)] = dict(ar=ar, tr=np.round(pr(S[n]), 3).tolist(), g=np.round(p3, 3).tolist(), q=np.round(q, 2).tolist(), rd=rd.astype(int).tolist(), view=np.round(vv, 4).tolist(), ticks=list(range(0, M, SEG)))
json.dump(out, open('count_rib3d.json', 'w'), separators=(',', ':')); print('var', out['var'], 'KB', len(json.dumps(out)) // 1024)
