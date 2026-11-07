"""Pretraining a counter with explicit cardinal supervision (stack 2 run 8, the positive control).
Streams of onset pulses on the blue channel (counted, 0-15) and the dark channel (ignored, 0-7) in random order; inter-onset
intervals 10-30 steps (as in a scan) or, with probability 0.15, 50-500 steps (1-10 s); a final silence of 0-500 steps. A 16-way
one-hot readout must report the running blue count at every step. Same 100 units, heterogeneous tau, noise and L2 as the stacks.
    python count_pre.py [steps=2000] [batch=128] [lr=1e-3] [tag=pre]
Saves count_w_TAG.npz: J, B (5 columns, only the first two used), c, W (16 x N), cz, alpha, pulse. Transfer to the game with
count_multi cfg {"init": "count_w_TAG.npz", "transfer": true} (J, B, c, alpha kept; readout fresh)."""
import os; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, time, numpy as np, torch, count as c, count_var as v, count_torch as ct, count_multi as cm
dev = ct.dev; KMAX = 15; NOUTP = KMAX + 1

def make_stream(rng, B, kmax=KMAX, tail=500):
    """U (B,T,5), Y (B,T,16) running-count one-hot, M (B,T) 1 inside the stream, meta: per-stream onsets and counts."""
    streams = []
    for i in range(B):
        nb = int(rng.integers(0, kmax + 1)); nd = int(rng.integers(0, kmax // 2 + 1))
        ev = np.array(['b'] * nb + ['d'] * nd); rng.shuffle(ev)
        ioi = np.where(rng.random(len(ev)) < 0.15, rng.integers(50, 501, len(ev)), rng.integers(10, 31, len(ev)))
        on = int(rng.integers(10, 16)) + np.concatenate([[0], np.cumsum(ioi[:-1])]) if len(ev) else np.zeros(0, int)
        L = (int(on[-1]) if len(ev) else 12) + v.PULSE_LEN + int(rng.integers(0, tail + 1))
        streams.append((ev, on, L))
    T = max(L for _, _, L in streams)
    U = np.zeros((B, T, c.NIN), np.float32); Y = np.zeros((B, T, NOUTP), np.float32); M = np.zeros((B, T), np.float32); meta = []
    for i, (ev, on, L) in enumerate(streams):
        count = np.zeros(L, int); k = 0
        for e, t in zip(ev, on):
            U[i, t:t + v.PULSE_LEN, 0 if e == 'b' else 1] = v.PULSE_AMP
            if e == 'b': k += 1; count[t:] = k
        Y[i, np.arange(L), count] = 1; M[i, :L] = 1
        meta.append(dict(ev=''.join(ev), on=on.tolist(), L=L, nb=int((ev == 'b').sum())))
    return U, Y, M, meta

def evaluate(Wt, alpha, seed=3):
    """Argmax accuracy at the last step before each onset (the count must be held across the gap), at the end of the final
    silence, and after the long (1-10 s) gaps only; plus accuracy by true count at the stream end."""
    U, Y, M, meta = make_stream(np.random.default_rng(seed), 256); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    with torch.no_grad():                                                   # 64 streams per forward: 256 at once needed ~3 GB at N = 300
        Z = np.concatenate([ct.forward(Wt, torch.tensor(U[s:s + 64], device=dev), True, False, gen, None, alpha)[0].argmax(2).cpu().numpy() for s in range(0, len(U), 64)], 0)
    Yc = Y.argmax(2); ok = tot = okl = totl = oke = 0; bycount = {}
    for i, m in enumerate(meta):
        for j, t in enumerate(m['on'][1:], 1):
            ok += Z[i, t - 1] == Yc[i, t - 1]; tot += 1
            if t - m['on'][j - 1] >= 50: okl += Z[i, t - 1] == Yc[i, t - 1]; totl += 1
        e = m['L'] - 1; oke += Z[i, e] == Yc[i, e]; bycount.setdefault(m['nb'], []).append(Z[i, e] == Yc[i, e])
    return dict(gap=round(ok / max(tot, 1), 3), long_gap=round(okl / max(totl, 1), 3), end=round(oke / len(meta), 3),
                by_count=[round(float(np.mean(bycount.get(k, [0]))), 2) for k in range(KMAX + 1)])

if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    steps = int(kw.get('steps', 2000)); batch = int(kw.get('batch', 128)); lr0 = float(kw.get('lr', 1e-3)); tag = kw.get('tag', 'pre')
    c.N = int(kw.get('N', 100)); tau_frac = float(kw.get('tau_frac', 1 / 3)); lo, hi = (float(x) for x in kw.get('tau_range', '1000,2000').split(','))
    rng = np.random.default_rng(1); W0 = c.init(rng); W0['W'] = np.zeros((NOUTP, c.N)); W0['cz'] = np.zeros(NOUTP)
    alpha = torch.tensor(cm.hetero_alpha(c.N, frac=tau_frac, lo=lo, hi=hi), dtype=torch.float32, device=dev)
    print(f'N {c.N} tau_frac {tau_frac} tau_range {lo},{hi}', flush=True)
    Wt = ct.to_t(W0); opt = torch.optim.Adam(Wt.values(), lr=lr0); gen = torch.Generator(device=dev); gen.manual_seed(5); t0 = time.time()
    print(tag, dev, f'batch {batch} steps {steps} lr {lr0} sig {c.SIG_REC},{c.SIG_IN} l2r {c.L2R} start', evaluate(Wt, alpha), flush=True)
    for it in range(1, steps + 1):
        lr = lr0 * (0.1 ** (it / steps))
        for g in opt.param_groups: g['lr'] = lr
        U, Y, M, meta = make_stream(rng, batch); U, Y, M = (torch.tensor(a, device=dev) for a in (U, Y, M))
        Z, R = ct.forward(Wt, U, True, True, gen, None, alpha)
        loss = (M[:, :, None] * (Z - Y) ** 2).sum() / (M.sum() * NOUTP) + c.L2R * (M[:, :, None] * R ** 2).sum() / (M.sum() * c.N) + c.L2W * (Wt['J'] ** 2).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(Wt.values()), 1.0); opt.step()
        if it % 50 == 0 or it == steps:
            print(it, round(loss.item(), 4), evaluate(Wt, alpha), f'{time.time() - t0:.0f}s', flush=True)
            np.savez(f'count_w_{tag}.npz', **ct.to_np(Wt), alpha=alpha.cpu().numpy().astype(np.float64), pulse=np.array(True))
    print('final', evaluate(Wt, alpha, seed=7), flush=True)
