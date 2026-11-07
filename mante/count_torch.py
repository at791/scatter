"""GPU port of the serial-counting training loop (same model, generators, loss and schedules as count.py / count_var.py).
    python count_torch.py VAR STEPS LR0 [fresh|INIT.npz] [tag=NAME] [batch=512] [loss=norm] [gain=1.5] [l2r=0] [sig=0.15,0.06] [gen=pause|fix|hold]
VAR as in count_var.py (base, big, digit, two, commit). Weights are saved as npz with the numpy keys so every analysis script still works.
Default batch 512 (the CPU scripts used 64), so steps are not comparable one-to-one with the CPU logs."""
import sys, time, numpy as np, torch, count as c, count_var as v
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(1)

def to_t(Wt): return {k: torch.tensor(np.asarray(w), dtype=torch.float32, device=dev, requires_grad=True) for k, w in Wt.items()}
def to_np(Wt): return {k: w.detach().cpu().numpy().astype(np.float64) for k, w in Wt.items()}

def act_fn(act): return torch.nn.functional.softplus if act == 'softplus' else torch.tanh

def forward(Wt, U, noise=True, keep=False, gen=None, in_mask=None, alpha=None, act='tanh', adapt=None):
    """U: (B,T,NIN) tensor. Returns Z (B,T,NOUT) and the rates R (B,T,N). Same dynamics as count.forward, with the input drive
    and the readout done as single batched matmuls and four kernels per time step (addmm, lerp, add noise, nonlinearity).
    alpha: scalar or per-unit (N,) tensor dt/tau (default c.ALPHA); act: 'tanh' or 'softplus'; adapt: (beta, g) adaptation or None."""
    B, T, _ = U.shape; a = c.ALPHA if alpha is None else alpha; n = Wt['J'].shape[0]; f = act_fn(act)
    av = torch.zeros(B, n, device=dev) if adapt else None
    sr = c.SIG_REC * np.sqrt(2 / c.ALPHA) if noise else 0; si = c.SIG_IN * np.sqrt(2 / c.ALPHA) if noise else 0
    if si:
        un = torch.randn(U.shape, device=dev, generator=gen) * si
        U = U + (un * in_mask if in_mask is not None else un)
    pre = U @ Wt['B'].T + Wt['c']                                            # (B,T,N) input drive for every step
    xn = torch.randn(B, T, n, device=dev, generator=gen) * (sr * a) if sr else None
    JT = Wt['J'].T; x = torch.zeros(B, n, device=dev); r = f(x); rs = []
    pres = pre.unbind(1); xns = xn.unbind(1) if xn is not None else None   # unbind: one stack kernel in backward, not T dense zero-fills
    for t in range(T):
        drive = torch.addmm(pres[t], r, JT)                                 # r J^T + B u + c
        if av is not None: drive = drive - adapt[1] * av
        x = torch.lerp(x, drive, a)                                         # x <- (1-a) x + a drive
        if xns is not None: x = x + xns[t]
        r = f(x); rs.append(r)
        if av is not None: av = torch.lerp(av, r, adapt[0])
    R = torch.stack(rs, 1)
    return R @ Wt['W'].T + Wt['cz'], R

def loss_std(Z, Y, M):
    """The objective count.grads actually descends: sum M (Z-Y)^2 / (sum M * NOUT)."""
    return (M[:, :, None] * (Z - Y) ** 2).sum() / (M.sum() * c.NOUT)

def loss_norm(Z, Y, meta, scol, thr=0.3):
    """count_var.grads_norm: per-trial normalised periods, hinge silence on columns scol, squared response error."""
    B, T, _ = Z.shape; Ws = np.zeros((B, T, 1), np.float32); Wr = np.zeros((B, T, 1), np.float32)
    for i, m in enumerate(meta):
        det, L = m['det'], m['L']; r0 = det + c.GRACE
        if det > 0: Ws[i, :det] = c.PRE_W / det
        if L > r0: Wr[i, r0:L] = c.RESP_W / (L - r0)
    Ws, Wr = torch.tensor(Ws, device=dev), torch.tensor(Wr, device=dev)
    Hs = torch.clamp(Z - thr, min=0) * scol
    return (Ws * Hs ** 2 + Wr * (Z - Y) ** 2).sum() / (B * c.NOUT)

def total_loss(Wt, U, Y, M, meta, mode, scol, gen, in_mask, keep_rates=True, alpha=None, act='tanh', adapt=None):
    Z, R = forward(Wt, U, True, True, gen, in_mask, alpha, act, adapt)
    base = loss_norm(Z, Y, meta, scol) if mode == 'norm' else loss_std(Z, Y, M)
    reg = c.L2R * (R ** 2).mean() + c.L2W * (Wt['J'] ** 2).sum()
    return base + reg, Z

EVAL_CHUNK = 500                                                            # trials per evaluation forward: 3000 at once needs GBs at N=300 / long trials

def forward_chunks(Wt, U, gen, in_mask, alpha, act, adapt):
    """No-grad forward over trials in chunks of EVAL_CHUNK; returns Z as numpy (B,T,NOUT)."""
    out = []
    with torch.no_grad():
        for s in range(0, U.shape[0], EVAL_CHUNK):
            Z, _ = forward(Wt, torch.tensor(U[s:s + EVAL_CHUNK], device=dev), True, False, gen, in_mask, alpha, act, adapt); out.append(Z.cpu().numpy())
    return np.concatenate(out, 0)

def evaluate(Wt, items, var, seed=3, in_mask=None, alpha=None, act='tanh', adapt=None):
    U, Y, M, meta = c.make(items, np.random.default_rng(0), interrupt=False); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    Z = forward_chunks(Wt, U, gen, in_mask, alpha, act, adapt); ok = early = never = 0; rts = []
    for i, m in enumerate(meta):
        d = v.decide(Z[i, :m['L']], var)
        if d is None: never += 1; continue
        if d[0] < m['det']: early += 1; continue
        if d[1] == m['lab']: ok += 1; rts.append((d[0] - m['det']) * 20)
    n = len(meta); return dict(acc=round(ok / n, 4), early=round(early / n, 4), never=round(never / n, 4), rt_ms=round(float(np.median(rts)), 0) if rts else None)

def bybl(Wt, items, var, in_mask=None, seed=0, alpha=None, act='tanh', adapt=None):
    U, Y, M, meta = c.make(items, np.random.default_rng(seed), interrupt=False); gen = torch.Generator(device=dev); gen.manual_seed(seed + 1)
    Z = forward_chunks(Wt, U, gen, in_mask, alpha, act, adapt); ok = np.zeros(len(meta), bool)
    for i, m in enumerate(meta):
        d = v.decide(Z[i, :m['L']], var)
        if d and d[0] >= m['det']: ok[i] = d[1] == m['lab']
    nb = np.array([m['st'].count('b') for m in meta]); lab = np.array([m['lab'] for m in meta])
    return (f'all {ok.mean():.3f} <=4 {ok[nb <= 4].mean():.3f} >4 {ok[nb > 4].mean():.3f} by label {[round(float(ok[lab == l].mean()), 2) for l in range(3)]} '
            f'by blues 0..10,11+ {[round(float(ok[nb == b].mean()), 2) for b in range(11)] + [round(float(ok[nb >= 11].mean()), 2)]}')

if __name__ == '__main__':
    var, steps, lr0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); rest = sys.argv[4:]
    fresh = 'fresh' in rest; init = next((a for a in rest if a.endswith('.npz')), None)
    kw = dict(a.split('=', 1) for a in rest if '=' in a)
    batch = int(kw.get('batch', 512)); mode = kw.get('loss', 'std'); gain = float(kw.get('gain', 1.0))
    if 'l2r' in kw: c.L2R = float(kw['l2r'])
    if 'sig' in kw: c.SIG_REC, c.SIG_IN = map(float, kw['sig'].split(','))
    tag = kw.get('tag', f'{var}_gpu')
    rng = np.random.default_rng(1 if fresh else 11); Wt = v.setup(var, fresh, rng, init)
    if fresh and gain != 1.0: Wt['J'] = Wt['J'] * gain
    if kw.get('gen') in ('pause', 'fix', 'hold'): c.make = v.make_gen(pause=kw['gen'] in ('pause', 'hold'), wide=kw['gen'] in ('fix', 'hold'))
    in_mask = None
    if var == 'digit':
        in_mask = torch.zeros(c.NIN, device=dev); in_mask[:5] = 1           # no sensory noise on the numeral channels
    scol = torch.ones(c.NOUT, device=dev)
    if var == 'commit': scol[:3] = 0                                        # silence is enforced on the commit unit only
    Wt = to_t(Wt); opt = torch.optim.Adam(Wt.values(), lr=lr0, betas=(0.9, 0.999), eps=1e-8)
    data = '../rnn/out_data.json'; test = c.real(data, 'TEST', np.random.default_rng(2))[:800]
    print(tag, dev, f'batch {batch} loss {mode} gain {gain} l2r {c.L2R} sig {c.SIG_REC},{c.SIG_IN} gen {kw.get("gen", "std")}', 'start', evaluate(Wt, test, var, in_mask=in_mask), flush=True)
    gen = torch.Generator(device=dev); gen.manual_seed(5); t0 = time.time()
    for it in range(1, steps + 1):
        if fresh:
            if it <= steps // 2: c.PRE_W = 1.0; lr = lr0; c.MAXSIZE = 8 if it <= steps * 3 // 16 else 14 if it <= steps * 5 // 16 else 21
            else: c.PRE_W = 3.0; lr = 0.3 * lr0 * (0.1 ** ((it - steps // 2) / max(1, steps - steps // 2))); c.MAXSIZE = 21
        else: lr = lr0 * (0.1 ** (it / steps)); c.MAXSIZE = 21
        for g in opt.param_groups: g['lr'] = lr
        items = c.synth(rng, batch); U, Y, M, meta = c.make(items, rng)
        U, Y, M = (torch.tensor(a, device=dev) for a in (U, Y, M))
        loss, _ = total_loss(Wt, U, Y, M, meta, mode, scol, gen, in_mask)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(Wt.values()), 1.0); opt.step()
        if it % 50 == 0 or it == steps:
            print(it, round(loss.item(), 4), evaluate(Wt, test, var, in_mask=in_mask), f'{time.time() - t0:.0f}s', flush=True)
            np.savez(f'count_w_{tag}.npz', **to_np(Wt))
    c.MAXSIZE = 21; c.PRE_W = 3.0
    print('final (training condition) real positions   ', bybl(Wt, c.real(data, 'TEST', np.random.default_rng(2))[:3000], var, in_mask), flush=True)
    print('final (training condition) balanced synthetic', bybl(Wt, c.synth_even(np.random.default_rng(99), 3000), var, in_mask), flush=True)
    c.make = v.make0 if var not in ('digit', 'two', 'commit') else c.make; c.SIG_REC, c.SIG_IN = 0.05, 0.02
    print('final (standard schedule+noise) real positions   ', bybl(Wt, c.real(data, 'TEST', np.random.default_rng(2))[:3000], var, in_mask), flush=True)
    print('final (standard schedule+noise) balanced synthetic', bybl(Wt, c.synth_even(np.random.default_rng(99), 3000), var, in_mask), flush=True)
