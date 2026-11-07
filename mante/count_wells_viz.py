"""What the wells look like: the pretrained cardinal counter (count_w_pre300.npz, number input off) beside the base game net
(count_w_w_base.npz, number 8 on). Noise off, per-unit time constants, no stones: x <- x + alpha (-x + J tanh x + h).
Everything is the network's own: the count states are where it settles after k blue pulses and 10 s of silence (the game
net: 1 s, since its line drifts), the plane is the PCA of those states, the background is log10 of the one-step speed
|F(x) - x|^2 on that plane, and the paths are where states dropped on the plane go in 10 s.
    python count_wells_viz.py OUT.json [preview.png]"""
import sys, json, base64, numpy as np, count as c, count_var as v
PUL, AMP, IOI, T10 = v.PULSE_LEN, v.PULSE_AMP, 20, 500

def load(path, num):
    W = dict(np.load(path)); a = np.asarray(W['alpha'], float) * np.ones(W['J'].shape[0])
    u = np.zeros(W['B'].shape[1]); u[3] = num / 10
    return dict(J=W['J'], h=W['B'] @ u + W['c'], a=a, B=W['B'], c=W['c'], u=u, N=W['J'].shape[0])

def step(M, X): return X + M['a'] * (-X + np.tanh(X) @ M['J'].T + M['h'])
def speed(M, X): d = M['a'] * (-X + np.tanh(X) @ M['J'].T + M['h']); return (d * d).sum(-1)

def count_states(M, K, settle):
    """State after k blue pulses (20-step spacing) and `settle` silent steps, from rest, for k = 0..K."""
    out = []
    for k in range(K + 1):
        x = np.zeros(M['N'])
        for t in range(10 + IOI * k):
            ub = M['u'].copy(); j = t - 10
            if 0 <= j and j % IOI < PUL and j // IOI < k: ub[0] = AMP
            x = x + M['a'] * (-x + np.tanh(x) @ M['J'].T + M['B'] @ ub + M['c'])
        for t in range(settle): x = step(M, x)
        out.append(x)
    return np.array(out)

def build(path, num, K, settle, tag, label):
    M = load(path, num); S = count_states(M, K, settle)
    mu = S.mean(0); U, s, Vt = np.linalg.svd(S - mu, full_matrices=False); P = Vt[:2]; var = (s ** 2 / (s ** 2).sum())[:3]
    C2 = (S - mu) @ P.T                                                           # count states on the plane
    lo, hi = C2.min(0), C2.max(0); pad = 0.18 * (hi - lo).max(); lo -= pad; hi += pad
    G = 110; ga = np.linspace(lo[0], hi[0], G); gb = np.linspace(lo[1], hi[1], G)
    A, Bm = np.meshgrid(ga, gb); X = mu + A.reshape(-1, 1) * P[0] + Bm.reshape(-1, 1) * P[1]
    q = np.log10(speed(M, X) + 1e-12).reshape(G, G)
    # paths of states dropped on a grid of the plane, 10 s
    na, nb = 14, 9; sa = np.linspace(lo[0] + pad * .5, hi[0] - pad * .5, na); sb = np.linspace(lo[1] + pad * .5, hi[1] - pad * .5, nb)
    A0, B0 = np.meshgrid(sa, sb); X = mu + A0.reshape(-1, 1) * P[0] + B0.reshape(-1, 1) * P[1]; tr = [X @ P.T * 0 + (X - mu) @ P.T]
    keep = [0, 5, 10, 25, 50, 100, 150, 200, 300, 400, 500]; path = []
    for t in range(T10 + 1):
        if t in keep: path.append((X - mu) @ P.T)
        if t < T10: X = step(M, X)
    end = X; dK = np.linalg.norm(end[:, None, :] - S[None], axis=2); kN = dK.argmin(1)
    stepsize = np.median(np.linalg.norm(np.diff(S, axis=0), axis=1)); conv = dK.min(1) / stepsize
    # speed along the chain of count states
    prof = []
    for k in range(K):
        t = np.linspace(0, 1, 21)[:-1] if k < K - 1 else np.linspace(0, 1, 21)
        Xs = S[k] + t[:, None] * (S[k + 1] - S[k]); prof += [[k + float(tt), float(np.log10(speed(M, x[None])[0] + 1e-12))] for tt, x in zip(t, Xs)]
    # half-kick starts between neighbouring counts, their 10 s paths
    hk = []
    for k in range(0, K, max(1, K // 5)):
        for f in (0.25, 0.5, 0.75):
            x = S[k] + f * (S[k + 1] - S[k]); pts = [(x - mu) @ P.T]
            for t in range(T10):
                x = step(M, x)
                if (t + 1) in keep: pts.append((x - mu) @ P.T)
            hk.append(dict(k=k, f=f, p=np.round(np.array(pts), 3).tolist()))
    J32 = M['J'].astype(np.float32)
    net = dict(tag=tag, label=label, number=num, K=K, settle_s=settle * 0.02, var=np.round(var, 3).tolist(),
               mu=np.round(mu, 5).tolist(), P=np.round(P, 6).tolist(), counts=np.round(C2, 4).tolist(),
               ext=[float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])], G=G, qmin=float(q.min()), qmax=float(q.max()),
               q=base64.b64encode(np.clip((q - q.min()) / (q.max() - q.min()) * 255, 0, 255).astype(np.uint8).tobytes()).decode(),
               paths=np.round(np.stack(path, 1), 3).tolist(), keep=keep, endK=kN.tolist(), conv=np.round(conv, 3).tolist(), profile=prof, halfkick=hk,
               J=base64.b64encode(J32.tobytes()).decode(), h=np.round(M['h'], 6).tolist(), alpha=np.round(M['a'], 6).tolist())
    print(f'{tag}: K {K}, plane var {np.round(var, 3)}, step size {stepsize:.2f}, dropped states ending within 0.25 step of a count state '
          f'{(conv < 0.25).mean():.2f}, distinct counts reached {len(set(kN[conv < 0.25].tolist()))}, speed at counts vs between: '
          f'{np.median([p[1] for p in prof if abs(p[0] - round(p[0])) < 1e-6]):.2f} / {np.median([p[1] for p in prof if abs(p[0] - round(p[0]) ) > 0.4]):.2f}', flush=True)
    return net

out = dict(nets=[build('count_w_pre300.npz', 0, 15, 500, 'pre300', 'Cardinal counter, before the game'),
                 build('count_w_succ300.npz', 0, 15, 500, 'succ300', 'Successor net, before the game'),
                 build('count_w_w_base.npz', 8, 8, 50, 'w_base', 'Base game network, number 8')])
json.dump(out, open(sys.argv[1], 'w'), separators=(',', ':'))
if len(sys.argv) > 2:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    for j, n in enumerate(out['nets']):
        q = np.frombuffer(base64.b64decode(n['q']), np.uint8).reshape(n['G'], n['G']); e = n['ext']
        a = ax[0, j]; a.imshow(q, origin='lower', extent=e, cmap='viridis', aspect='auto')
        for p in n['paths']: p = np.array(p); a.plot(p[:, 0], p[:, 1], 'w-', lw=.6, alpha=.7); a.plot(p[-1, 0], p[-1, 1], 'o', ms=2.5, color='w')
        C = np.array(n['counts']); a.plot(C[:, 0], C[:, 1], 'o', ms=7, mfc='none', mec='r', mew=1.5)
        for i, (x, y) in enumerate(C): a.text(x, y, str(i), color='r', fontsize=9)
        a.set_title(f"{n['label']} (plane var {n['var'][:2]})")
        b = ax[1, j]; P = np.array(n['profile']); b.plot(P[:, 0], P[:, 1]); b.set_xlabel('position along the chain of count states'); b.set_ylabel('log10 speed')
    plt.tight_layout(); plt.savefig(sys.argv[2], dpi=80)
