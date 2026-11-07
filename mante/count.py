"""Serial counting with a free response. Mante-style rate RNN (100 tanh units, Mante init), tau = 100 ms, dt = 20 ms,
noise in units and inputs, light L2. One scripted pass over the circle: each stone fixated once (200-300 ms), gaps 0-300 ms,
occasional long interruptions. Inputs: fixated stone (blue/dark/hidden), number (tonic), scan-complete (tonic after last stone).
Outputs: 3 units (rest dark / rest blue / can't tell). Targets: 0 until the verdict is determined, then one-hot, with a grace
period so it may answer when ready. 'Rest dark' is determined as soon as blues seen == number; the others when the scan is done."""
import sys, json, time, numpy as np
N, ALPHA, NIN, NOUT = 100, 0.2, 5, 3          # alpha = dt/tau = 20/100
SIG_REC, SIG_IN, L2R, L2W = 0.05, 0.02, 1e-4, 1e-5
GRACE, POST = 15, 50                          # steps: grace after determination; response period after scan complete
RESP_W = 4.0                                  # loss weight on the response period (it is a small share of each trial)
PRE_W = 3.0                                   # loss weight on staying silent before the verdict is determined (early answers are errors)
MAXSIZE = 21                                  # curriculum: largest circle used by synth(); raised during training
NUMMAX = 15                                   # largest number cue / visible blue count used by synth() and synth_even() (count curriculum lowers it)

def label(num, nb, nh):
    if nh == 0: return None
    return 0 if num == nb else 1 if num - nb == nh else 2

def schedule(st, rng, interrupt=True):
    """Returns per-step fixated stone index (-1 = none) and the step at which the scan completes."""
    seq = []
    seq += [-1] * int(rng.integers(10, 16))                                   # number only, 200-300 ms
    for j in range(len(st)):
        seq += [j] * int(rng.integers(10, 16))                                # fixation 200-300 ms
        gap = int(rng.integers(0, 16))                                        # 0-300 ms
        if interrupt and rng.random() < 0.08: gap += int(rng.integers(30, 61))  # 600-1200 ms interruption
        seq += [-1] * gap
    done = len(seq); seq += [-1] * POST
    return np.array(seq), done

def make(items, rng, interrupt=True):
    """items: list of (num, states). Returns U (B,T,NIN), Y (B,T,NOUT), M (B,T) loss mask, meta."""
    scheds = [schedule(st, rng, interrupt) for _, st in items]
    T = max(len(s) for s, _ in scheds); B = len(items)
    U = np.zeros((B, T, NIN), np.float32); Y = np.zeros((B, T, NOUT), np.float32); M = np.zeros((B, T), np.float32); meta = []
    for i, ((num, st), (seq, done)) in enumerate(zip(items, scheds)):
        L = len(seq); U[i, :L, 3] = num / 10; U[i, done:L, 4] = 1
        nb = 0; det = None; lab = label(num, st.count('b'), st.count('h'))
        for t in range(L):
            j = seq[t]
            if j >= 0:
                U[i, t, 'bdh'.index(st[j])] = 1
                if t == 0 or seq[t - 1] != j:
                    if st[j] == 'b': nb += 1
                    if lab == 0 and nb == num and det is None: det = t + 1   # rest dark determined at this fixation
        if det is None: det = done
        Y[i, det:L, lab] = 1
        M[i, :det] = PRE_W; M[i, det + GRACE:L] = RESP_W                           # grace: no loss for GRACE steps after determination; response period weighted
        meta.append(dict(num=num, st=''.join(st), lab=lab, det=int(det), done=int(done), L=int(L), idx=seq.tolist()))
    return U, Y, M, meta

def synth(rng, n):
    items = []; want = np.tile([0, 1, 2], n // 3 + 1)[:n]; k = 0
    while k < n:
        size = int(rng.integers(2, MAXSIZE + 1)); num = int(rng.integers(0, min(size, NUMMAX) + 1))
        blue = np.zeros(size, bool); blue[rng.choice(size, num, replace=False)] = True
        hid = rng.random(size) < rng.random()
        lab = label(num, int((blue & ~hid).sum()), int(hid.sum()))
        if lab != want[k]: continue
        items.append((num, ['h' if hid[i] else 'b' if blue[i] else 'd' for i in range(size)])); k += 1
    return items

def synth_even(rng, n, vmax=None):
    """Balanced generator: visible blue count uniform on 0..vmax (default NUMMAX), verdicts cycled 0/1/2, size uniform 2..MAXSIZE, number <= NUMMAX."""
    if vmax is None: vmax = NUMMAX
    items = []; want = np.tile([0, 1, 2], n // 3 + 1)[:n]; k = 0
    while k < n:
        v = int(rng.integers(0, vmax + 1)); lab = want[k]
        size = int(rng.integers(max(2, v + 1), MAXSIZE + 1)); nh = int(rng.integers(1, size - v + 1))
        if lab == 0: need = 0
        elif lab == 1: need = nh
        else:
            if nh < 2: continue
            need = int(rng.integers(1, nh))
        num = v + need
        if num > NUMMAX: continue
        st = ['b'] * v + ['h'] * nh + ['d'] * (size - v - nh); rng.shuffle(st)
        assert label(num, v, nh) == lab
        items.append((num, st)); k += 1
    return items

def real(path, split, rng):
    d = json.load(open(path))[split]; F = 10; out = []
    for g in d:
        X = g['X']; mem = {}
        for s, t in zip(g['s0'], g['d0']): mem.setdefault(s, []).append(t)
        for k, m in mem.items():
            num = round(X[k * F + 4] * 10)
            st = ['h' if X[j * F] == 1 else 'b' if X[j * F + 1] == 1 else 'd' for j in m]
            if label(num, st.count('b'), st.count('h')) is None: continue
            rng.shuffle(st); out.append((num, st))
    return out

def init(rng):
    return dict(J=rng.normal(0, 1 / np.sqrt(N), (N, N)), B=rng.normal(0, 0.5, (N, NIN)), c=np.zeros(N), W=np.zeros((NOUT, N)), cz=np.zeros(NOUT))

def act_of(Wt): return str(Wt['act']) if 'act' in Wt else 'tanh'
def phi(x, Wt):
    """Unit nonlinearity: tanh (default) or softplus (unbounded rates) when the checkpoint carries act='softplus'."""
    return np.logaddexp(0, x) if act_of(Wt) == 'softplus' else np.tanh(x)
def dphi(x, Wt):
    return 1 / (1 + np.exp(-x)) if act_of(Wt) == 'softplus' else 1 - np.tanh(x) ** 2
def alpha_of(Wt):
    """dt/tau, a scalar or a per-unit vector (checkpoints with heterogeneous time constants carry 'alpha')."""
    return np.asarray(Wt['alpha'], dtype=float) if 'alpha' in Wt else ALPHA

def adapt_of(Wt):
    """Per-unit adaptation (beta = dt/tau_a, strength g): a slow trace of each unit's rate subtracted from its drive.
    Present when the checkpoint carries 'adapt_g'; else None."""
    return (float(Wt['adapt_beta']), float(Wt['adapt_g'])) if 'adapt_g' in Wt else None

def forward(Wt, U, rng=None, noise=True, keep=False):
    """Checkpoints may carry per-unit 'alpha', an 'act' name and adaptation; grads() below stays tanh / scalar-alpha only."""
    B, T, _ = U.shape; n = Wt['J'].shape[0]; x = np.zeros((B, n)); xs = [x]; zs = []; us = []; a = alpha_of(Wt)
    ad = adapt_of(Wt); av = np.zeros((B, n))
    sr = SIG_REC * np.sqrt(2 / ALPHA) if noise else 0; si = SIG_IN * np.sqrt(2 / ALPHA) if noise else 0
    for t in range(T):
        r = phi(x, Wt); u = U[:, t] + (si * rng.standard_normal(U[:, t].shape) if si else 0)
        drive = r @ Wt['J'].T + u @ Wt['B'].T + Wt['c']
        if ad: drive = drive - ad[1] * av
        x = (1 - a) * x + a * drive + (sr * a * rng.standard_normal(x.shape) if sr else 0)
        rn = phi(x, Wt); z = rn @ Wt['W'].T + Wt['cz']; zs.append(z)
        if ad: av = (1 - ad[0]) * av + ad[0] * rn
        if keep: xs.append(x); us.append(u)
    return np.stack(zs, 1), (xs, us) if keep else None

def grads(Wt, U, Y, M, rng):
    B, T, _ = U.shape; Z, (xs, us) = forward(Wt, U, rng, True, True)
    E = (Z - Y) * M[:, :, None]; nm = M.sum() * NOUT; loss = float((E ** 2).sum() / nm)
    G = {k: np.zeros_like(v) for k, v in Wt.items()}
    rs = [np.tanh(x) for x in xs]
    for t in range(T):
        le = 2 * E[:, t] / nm; G['W'] += le.T @ rs[t + 1]; G['cz'] += le.sum(0)
    loss += L2R * float(np.mean([np.mean(r ** 2) for r in rs[1:]])) + L2W * float((Wt['J'] ** 2).sum())
    gx = np.zeros((B, N))
    for t in range(T - 1, -1, -1):
        le = 2 * E[:, t] / nm; r1 = rs[t + 1]
        gr1 = le @ Wt['W'] + 2 * L2R * r1 / (T * B * N)                        # dL/dr_{t+1} from readout and L2 on rates
        gx = gx + gr1 * (1 - r1 ** 2)
        r0 = rs[t]
        G['J'] += ALPHA * gx.T @ r0; G['B'] += ALPHA * gx.T @ us[t]; G['c'] += ALPHA * gx.sum(0)
        gr0 = ALPHA * gx @ Wt['J']
        gx = (1 - ALPHA) * gx + gr0 * (1 - r0 ** 2)
    G['J'] += 2 * L2W * Wt['J']
    return loss, G

def evaluate(Wt, items, rng, thr=0.5):
    U, Y, M, meta = make(items, np.random.default_rng(0), interrupt=False); Z, _ = forward(Wt, U, rng, noise=True)
    ok = 0; early = 0; rts = []; never = 0
    for i, m in enumerate(meta):
        z = Z[i, :m['L']]; above = np.where(z.max(1) > thr)[0]
        if len(above) == 0: never += 1; continue
        t0 = above[0]
        if t0 < m['det']: early += 1; continue
        choice = int(z[t0].argmax())
        if choice == m['lab']: ok += 1; rts.append((t0 - m['det']) * 20)
    n = len(meta); return dict(acc=round(ok / n, 4), early=round(early / n, 4), never=round(never / n, 4), rt_ms=round(float(np.median(rts)), 0) if rts else None)

if __name__ == '__main__':
    steps, data = int(sys.argv[1]), sys.argv[2]; lr0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    rng = np.random.default_rng(1); Wt = init(rng)
    if len(sys.argv) > 4: Wt = dict(np.load(sys.argv[4])); rng = np.random.default_rng(7)
    test = real(data, 'TEST', np.random.default_rng(2))[:800]
    m = {k: np.zeros_like(v) for k, v in Wt.items()}; v2 = {k: np.zeros_like(v) for k, v in Wt.items()}; t0 = time.time()
    for it in range(1, steps + 1):
        lr = lr0 * (0.1 ** (it / steps)) if len(sys.argv) > 4 else lr0
        MAXSIZE = (8 if it <= 1500 else 14 if it <= 2500 else 21) if len(sys.argv) <= 4 else 21   # curriculum on circle size (fresh runs only)
        items = (synth_even if len(sys.argv) > 6 and sys.argv[6] == 'even' else synth)(rng, 64); U, Y, M, _ = make(items, rng)
        loss, G = grads(Wt, U, Y, M, rng)
        gn = np.sqrt(sum((g ** 2).sum() for g in G.values())); sc = min(1, 1 / (gn + 1e-9))
        for k in Wt:
            g = G[k] * sc; m[k] = 0.9 * m[k] + 0.1 * g; v2[k] = 0.999 * v2[k] + 0.001 * g * g
            Wt[k] -= lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v2[k] / (1 - 0.999 ** it)) + 1e-8)
        if it % 100 == 0 or it == steps:
            print(it, round(loss, 4), evaluate(Wt, test, np.random.default_rng(3)), f'{time.time() - t0:.0f}s', flush=True)
            np.savez(sys.argv[5] if len(sys.argv) > 5 else 'count_w.npz', **Wt)
