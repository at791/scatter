"""Pretraining on the successor function (stack 7; user, 2026-11-02: "add a network pretrained on the successor function to the
pipeline"). Same streams, units, time constants, noise, L2 and seeds as count_pre.py (the cardinal counter), so the two nets start
from identical weights and see identical data; only the target differs. Instead of a one-hot label per count, the readout must hold
a 4-D code that one fixed rotation R advances at every blue: v_k = R^k v_0, v_k = (cos k pi/8, sin k pi/8, cos 3k pi/8, sin 3k pi/8).
The net is taught the step "next", the same operator at every count, not names for the counts (the successor route of Dulberg,
Webb and Cohen 2021). The 16 codes are distinct and at least 1.18 apart (one-hot labels: 1.41). The loss keeps count_pre's
normalisation (summed squared error / 16 per step), so the regularisers keep their relative weight. Evaluation decodes the nearest
code and reuses count_pre.evaluate, so its numbers compare directly with the counter's.
    python count_succ.py [steps=2000] [batch=128] [lr=1e-3] [tag=succ] [N=100] [tau_frac=0.333] [tau_range=1000,2000]
Saves count_w_TAG.npz: J, B (5 columns, only the first two used), c, W (4 x N), cz, alpha, pulse, succ (the code's angles).
Transfer to the game like a count_pre checkpoint (count_multi cfg {"init": "count_w_TAG.npz", "transfer": true});
count_halfkick.py tests it at number 0, as a counter."""
import os; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, time, numpy as np, torch, count as c, count_torch as ct, count_multi as cm, count_pre as cp
dev = ct.dev; THETA = (np.pi / 8, 3 * np.pi / 8)
k = np.arange(cp.KMAX + 1); V = np.stack([f(k * th) for th in THETA for f in (np.cos, np.sin)], 1)   # (16, 4): row k is v_k
D = V.shape[1]; Vt = torch.tensor(V, dtype=torch.float32, device=dev)

def decoded(Wt):
    """The 16-way nearest-code readout: every v_k has norm sqrt(2), so the nearest code is the largest V z."""
    return {**Wt, 'W': Vt @ Wt['W'], 'cz': Vt @ Wt['cz']}

if __name__ == '__main__':
    kw = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    steps = int(kw.get('steps', 2000)); batch = int(kw.get('batch', 128)); lr0 = float(kw.get('lr', 1e-3)); tag = kw.get('tag', 'succ')
    c.N = int(kw.get('N', 100)); tau_frac = float(kw.get('tau_frac', 1 / 3)); lo, hi = (float(x) for x in kw.get('tau_range', '1000,2000').split(','))
    rng = np.random.default_rng(1); W0 = c.init(rng); W0['W'] = np.zeros((D, c.N)); W0['cz'] = np.zeros(D)
    alpha = torch.tensor(cm.hetero_alpha(c.N, frac=tau_frac, lo=lo, hi=hi), dtype=torch.float32, device=dev)
    print(f'N {c.N} tau_frac {tau_frac} tau_range {lo},{hi} code angles {[round(t, 4) for t in THETA]}', flush=True)
    Wt = ct.to_t(W0); opt = torch.optim.Adam(Wt.values(), lr=lr0); gen = torch.Generator(device=dev); gen.manual_seed(5); t0 = time.time()
    with torch.no_grad(): ev = cp.evaluate(decoded(Wt), alpha)
    print(tag, dev, f'batch {batch} steps {steps} lr {lr0} sig {c.SIG_REC},{c.SIG_IN} l2r {c.L2R} start', ev, flush=True)
    for it in range(1, steps + 1):
        lr = lr0 * (0.1 ** (it / steps))
        for g in opt.param_groups: g['lr'] = lr
        U, Y, M, meta = cp.make_stream(rng, batch); Y = (Y @ V).astype(np.float32)   # one-hot running count -> its code v_k
        U, Y, M = (torch.tensor(a, device=dev) for a in (U, Y, M))
        Z, R = ct.forward(Wt, U, True, True, gen, None, alpha)
        loss = (M[:, :, None] * (Z - Y) ** 2).sum() / (M.sum() * cp.NOUTP) + c.L2R * (M[:, :, None] * R ** 2).sum() / (M.sum() * c.N) + c.L2W * (Wt['J'] ** 2).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(Wt.values()), 1.0); opt.step()
        if it % 50 == 0 or it == steps:
            with torch.no_grad(): ev = cp.evaluate(decoded(Wt), alpha)
            print(it, round(loss.item(), 4), ev, f'{time.time() - t0:.0f}s', flush=True)
            np.savez(f'count_w_{tag}.npz', **ct.to_np(Wt), alpha=alpha.cpu().numpy().astype(np.float64), pulse=np.array(True), succ=np.array(THETA))
    with torch.no_grad(): ev = cp.evaluate(decoded(Wt), alpha, seed=7)
    print('final', ev, flush=True)
