"""Variants of the serial-counting net, none adding a count output.   python count_var.py VAR STEPS LR0 [fresh]
base:  the plain 100-unit net (only meaningful with 'fresh', as a same-recipe baseline).
big:   100 fresh units appended (N=200). Resumed: old block kept, new block Mante-random, cross blocks small, new units unread at the start.
two:   the scan goes around the circle twice in the same order; a tonic 'second pass' input marks the repeat; verdict after the second pass.
digit: the number arrives as a one-hot numeral (16 channels) instead of a scalar; no input noise on the numeral channels.
       Resumed: input weights initialised so the drive is identical at the start.
Resumed runs start from count_w_skewed.npz with a decaying lr. 'fresh' runs start from a random init with the size curriculum
(8 -> 14 -> 21 stones at steps 1500/2500), constant lr for the first 4000 steps, then lr decaying 10x over the rest.
Training uses the same (skewed) generator as the kept network; the held-out test is the same 800 real positions."""
import sys, time, numpy as np, count as c
make0, forward0 = c.make, c.forward

def forward_digit(Wt, U, rng=None, noise=True, keep=False):
    """count.forward with input noise on the stone/scan channels only: a numeral is read without sensory noise."""
    Bn, T, _ = U.shape; x = np.zeros((Bn, c.N)); xs = [x]; zs = []; us = []; msk = np.zeros(c.NIN); msk[:5] = 1
    sr = c.SIG_REC * np.sqrt(2 / c.ALPHA) if noise else 0; si = c.SIG_IN * np.sqrt(2 / c.ALPHA) if noise else 0
    for t in range(T):
        r = np.tanh(x); u = U[:, t] + (si * msk * rng.standard_normal(U[:, t].shape) if si else 0)
        x = (1 - c.ALPHA) * x + c.ALPHA * (r @ Wt['J'].T + u @ Wt['B'].T + Wt['c']) + (sr * c.ALPHA * rng.standard_normal(x.shape) if sr else 0)
        z = np.tanh(x) @ Wt['W'].T + Wt['cz']; zs.append(z)
        if keep: xs.append(x); us.append(u)
    return np.stack(zs, 1), (xs, us) if keep else None

def make_digit(items, rng, interrupt=True):
    U, Y, M, meta = make0(items, rng, interrupt); U[:, :, 3] = 0
    for i, m in enumerate(meta): U[i, :m['L'], 5 + m['num']] = 1
    return U, Y, M, meta

def sched2(st, rng, interrupt):
    seq = [-1] * int(rng.integers(10, 16)); p2 = None
    for p in range(2):
        if p == 1: p2 = len(seq)
        for j in range(len(st)):
            seq += [j] * int(rng.integers(10, 16)); gap = int(rng.integers(0, 16))
            if interrupt and rng.random() < 0.08: gap += int(rng.integers(30, 61))
            seq += [-1] * gap
    done = len(seq); seq += [-1] * c.POST
    return np.array(seq), p2, done

def make_two(items, rng, interrupt=True):
    scheds = [sched2(st, rng, interrupt) for _, st in items]; T = max(len(s) for s, _, _ in scheds); Bn = len(items)
    U = np.zeros((Bn, T, c.NIN), np.float32); Y = np.zeros((Bn, T, c.NOUT), np.float32); M = np.zeros((Bn, T), np.float32); meta = []
    for i, ((num, st), (seq, p2, done)) in enumerate(zip(items, scheds)):
        L = len(seq); U[i, :L, 3] = num / 10; U[i, done:L, 4] = 1; U[i, p2:L, 5] = 1
        nb = 0; det = None; lab = c.label(num, st.count('b'), st.count('h'))
        for t in range(L):
            j = seq[t]
            if j >= 0:
                U[i, t, 'bdh'.index(st[j])] = 1
                if (t == 0 or seq[t - 1] != j) and st[j] == 'b':
                    nb += 1
                    if lab == 0 and nb == num and det is None: det = t + 1
        if det is None: det = done
        Y[i, det:L, lab] = 1; M[i, :det] = c.PRE_W; M[i, det + c.GRACE:L] = c.RESP_W
        meta.append(dict(num=num, st=''.join(st), lab=lab, det=int(det), done=int(done), L=int(L), p2=int(p2), idx=seq.tolist()))
    return U, Y, M, meta

def make_commit(items, rng, interrupt=True):
    """Fourth output = commit unit: 0 until the verdict is determined, 1 afterwards. Verdict units get no target during the scan."""
    U, Y, M, meta = make0(items, rng, interrupt)
    for i, m in enumerate(meta): Y[i, m['det']:m['L'], 3] = 1
    return U, Y, M, meta

PULSE_LEN, PULSE_AMP = 3, 2.0                                               # event input: 60 ms onset pulse, amplitude 2

def pulsify(U, meta):
    """Replace each sustained stone input by an onset pulse (PULSE_LEN steps, PULSE_AMP) at the start of the fixation, in place.
    The schedule, targets and masks are unchanged, so the only thing removed is dwell-time information."""
    for i, m in enumerate(meta):
        seq = m['idx']; L = m['L']; ch = U[i, :L, :3].argmax(1).copy(); U[i, :L, :3] = 0
        for t in range(L):
            if seq[t] >= 0 and (t == 0 or seq[t - 1] != seq[t]): U[i, t:t + PULSE_LEN, ch[t]] = PULSE_AMP
    return U

PAUSE_MAX = 0                                                               # pause-curriculum ceiling in steps; the trainer raises it
PAUSE_W = 15                                                                # a curriculum pause weighs like PAUSE_W silent steps in the loss

def add_pos(U, meta):
    """Append two channels: cos and sin of the fixated stone's angle round the circle (zero when nothing is fixated)."""
    B, T, _ = U.shape; P = np.zeros((B, T, 2), np.float32)
    for i, m in enumerate(meta):
        seq = np.array(m['idx']); n = len(m['st']); on = seq >= 0; ang = 2 * np.pi * seq[on] / n
        P[i, :len(seq)][on] = np.stack([np.cos(ang), np.sin(ang)], 1)
    return np.concatenate([U, P], 2)

def make_gen(pause=False, wide=False, pulse=False, curric=False, pos=False):
    """Generators that punish a leaky line. pause: a 2-8 s pause at a random point in every trial. wide: fixations 100-600 ms
    instead of 200-300 ms. Gaps stay 0-300 ms with the usual 8% long interruptions. pulse: stones as onset pulses (see pulsify).
    curric: a pause of 0..PAUSE_MAX steps at a random stone (marked -2 in idx), weighed like PAUSE_W silent steps so long
    silences do not swamp the loss. pos: append the fixated stone's angle as cos/sin input channels."""
    def mk(items, rng, interrupt=True):
        old = c.schedule
        def sched(st, rng, interrupt=True):
            seq = [-1] * int(rng.integers(10, 16)); where = int(rng.integers(0, len(st)))
            for j in range(len(st)):
                seq += [j] * int(rng.integers(5, 31) if wide else rng.integers(10, 16)); gap = int(rng.integers(0, 16))
                if interrupt and rng.random() < 0.08: gap += int(rng.integers(30, 61))
                if pause and j == where: gap += int(rng.integers(100, 401))
                seq += [-1] * gap
                if curric and j == where and PAUSE_MAX > 0: seq += [-2] * int(rng.integers(0, PAUSE_MAX + 1))
            done = len(seq); seq += [-1] * c.POST
            return np.array(seq), done
        if pause or wide or curric: c.schedule = sched
        nin0 = c.NIN
        if pos: c.NIN = nin0 - 2                                              # the base generator fills the stone/number/done channels only
        try: U, Y, M, meta = make0(items, rng, interrupt)
        finally: c.schedule = old; c.NIN = nin0
        if curric:
            for i, m in enumerate(meta):
                p = np.array(m['idx']) == -2; k = int(p.sum())
                if k > PAUSE_W: M[i, :len(p)][p] *= PAUSE_W / k
        if pulse: pulsify(U, meta)
        if pos: U = add_pos(U, meta)
        return U, Y, M, meta
    return mk

def grads_norm(Wt, U, Y, meta, rng, scol, thr=0.3):
    """Items 1+2: per-period normalised loss. In each trial the silent steps share weight PRE_W and the response steps share RESP_W,
    whatever the trial length; silence is a hinge (only output above thr is charged) on the columns in scol; the response loss is squared."""
    B, T, _ = U.shape; Z, (xs, us) = c.forward(Wt, U, rng, True, True)
    Ws = np.zeros((B, T, 1)); Wr = np.zeros((B, T, 1))
    for i, m in enumerate(meta):
        det, L = m['det'], m['L']; r0 = det + c.GRACE
        if det > 0: Ws[i, :det] = c.PRE_W / det
        if L > r0: Wr[i, r0:L] = c.RESP_W / (L - r0)
    Hs = np.maximum(Z - thr, 0) * scol; E = Ws * Hs + Wr * (Z - Y); nm = B * c.NOUT
    loss = float((Ws * Hs ** 2 + Wr * (Z - Y) ** 2).sum() / nm)
    G = {k: np.zeros_like(v) for k, v in Wt.items()}; rs = [np.tanh(x) for x in xs]
    for t in range(T):
        le = 2 * E[:, t] / nm; G['W'] += le.T @ rs[t + 1]; G['cz'] += le.sum(0)
    loss += c.L2R * float(np.mean([np.mean(r ** 2) for r in rs[1:]])) + c.L2W * float((Wt['J'] ** 2).sum())
    gx = np.zeros((B, c.N))
    for t in range(T - 1, -1, -1):
        le = 2 * E[:, t] / nm; r1 = rs[t + 1]
        gr1 = le @ Wt['W'] + 2 * c.L2R * r1 / (T * B * c.N)
        gx = gx + gr1 * (1 - r1 ** 2); r0 = rs[t]
        G['J'] += c.ALPHA * gx.T @ r0; G['B'] += c.ALPHA * gx.T @ us[t]; G['c'] += c.ALPHA * gx.sum(0)
        gx = (1 - c.ALPHA) * gx + (c.ALPHA * gx @ Wt['J']) * (1 - r0 ** 2)
    G['J'] += 2 * c.L2W * Wt['J']
    return loss, G

def decide(z, var):
    """First step at which the net commits, and its choice; None if it never commits."""
    if var == 'commit':
        ab = np.where(z[:, 3] > 0.5)[0]
        return (int(ab[0]), int(z[ab[0], :3].argmax())) if len(ab) else None
    ab = np.where(z.max(1) > 0.5)[0]
    return (int(ab[0]), int(z[ab[0]].argmax())) if len(ab) else None

def evaluate(Wt, items, rng, var):
    U, Y, M, meta = c.make(items, np.random.default_rng(0), interrupt=False); Z, _ = c.forward(Wt, U, rng, noise=True)
    ok = early = never = 0; rts = []
    for i, m in enumerate(meta):
        d = decide(Z[i, :m['L']], var)
        if d is None: never += 1; continue
        if d[0] < m['det']: early += 1; continue
        if d[1] == m['lab']: ok += 1; rts.append((d[0] - m['det']) * 20)
    n = len(meta); return dict(acc=round(ok / n, 4), early=round(early / n, 4), never=round(never / n, 4), rt_ms=round(float(np.median(rts)), 0) if rts else None)

def setup(var, fresh, rng, init=None):
    """Sets the module dimensions, patches count.make/forward, returns the starting weights.
    init: a same-variant checkpoint to resume from; otherwise the skewed checkpoint is widened to the variant."""
    c.make, c.forward = make0, forward0
    if var == 'big': c.N = 200
    if var == 'digit': c.NIN = 5 + 16; c.make, c.forward = make_digit, forward_digit
    if var == 'two': c.NIN = 6; c.make = make_two
    if var == 'commit': c.NOUT = 4; c.make = make_commit
    if fresh: return c.init(rng)
    if init: return dict(np.load(init))
    Wt = dict(np.load('count_w_skewed.npz'))
    if var == 'big':
        N0 = 100; n1 = c.N - N0
        J = np.zeros((c.N, c.N)); J[:N0, :N0] = Wt['J']; J[N0:, N0:] = rng.normal(0, 1 / np.sqrt(c.N), (n1, n1))
        J[:N0, N0:] = rng.normal(0, 0.1 / np.sqrt(c.N), (N0, n1)); J[N0:, :N0] = rng.normal(0, 0.1 / np.sqrt(c.N), (n1, N0))
        Wt = dict(J=J, B=np.vstack([Wt['B'], rng.normal(0, 0.1, (n1, c.NIN))]), c=np.concatenate([Wt['c'], np.zeros(n1)]),
                  W=np.hstack([Wt['W'], np.zeros((c.NOUT, n1))]), cz=Wt['cz'])
    elif var == 'digit':
        B = np.zeros((c.N, c.NIN)); B[:, :5] = Wt['B']; B[:, 3] = 0
        for n in range(16): B[:, 5 + n] = Wt['B'][:, 3] * n / 10
        Wt['B'] = B
    elif var == 'two': Wt['B'] = np.hstack([Wt['B'], np.zeros((c.N, 1))])
    return Wt

def bybl(Wt, items, seed=0):
    U, Y, M, meta = c.make(items, np.random.default_rng(seed), interrupt=False); Z, _ = c.forward(Wt, U, np.random.default_rng(seed + 1), True)
    ok = np.zeros(len(meta), bool)
    for i, m in enumerate(meta):
        z = Z[i, :m['L']]; ab = np.where(z.max(1) > 0.5)[0]
        if len(ab) and ab[0] >= m['det']: ok[i] = int(z[ab[0]].argmax()) == m['lab']
    nb = np.array([m['st'].count('b') for m in meta]); lab = np.array([m['lab'] for m in meta])
    return (f'all {ok.mean():.3f} <=4 {ok[nb <= 4].mean():.3f} >4 {ok[nb > 4].mean():.3f} by label {[round(float(ok[lab == l].mean()), 2) for l in range(3)]} '
            f'by blues 0..10,11+ {[round(float(ok[nb == b].mean()), 2) for b in range(11)] + [round(float(ok[nb >= 11].mean()), 2)]}')

if __name__ == '__main__':
    var, steps, lr0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); fresh = len(sys.argv) > 4 and sys.argv[4] == 'fresh'
    init = sys.argv[4] if len(sys.argv) > 4 and not fresh else None            # python count_var.py VAR STEPS LR0 count_w_big_fresh.npz [TAG]
    rng = np.random.default_rng(1 if fresh else 11); Wt = setup(var, fresh, rng, init)
    tag = sys.argv[5] if len(sys.argv) > 5 else (f'{var}_fresh2' if fresh else var)
    data = '../rnn/out_data.json'; test = c.real(data, 'TEST', np.random.default_rng(2))[:800]
    print(tag, 'start', c.evaluate(Wt, test, np.random.default_rng(3)), flush=True)
    m = {k: np.zeros_like(v) for k, v in Wt.items()}; v2 = {k: np.zeros_like(v) for k, v in Wt.items()}; t0 = time.time()
    for it in range(1, steps + 1):
        if fresh:   # staged like the kept net's history: learn to answer first (silent weight 1), then learn to wait (weight 3, decaying lr)
            if it <= 4000: c.PRE_W = 1.0; lr = lr0; c.MAXSIZE = 8 if it <= 1500 else 14 if it <= 2500 else 21
            else: c.PRE_W = 3.0; lr = 0.3 * lr0 * (0.1 ** ((it - 4000) / max(1, steps - 4000))); c.MAXSIZE = 21
        else: lr = lr0 * (0.1 ** (it / steps)); c.MAXSIZE = 21
        items = c.synth(rng, 64); U, Y, M, _ = c.make(items, rng)
        loss, G = c.grads(Wt, U, Y, M, rng)
        gn = np.sqrt(sum((g ** 2).sum() for g in G.values())); sc = min(1, 1 / (gn + 1e-9))
        for k in Wt:
            g = G[k] * sc; m[k] = 0.9 * m[k] + 0.1 * g; v2[k] = 0.999 * v2[k] + 0.001 * g * g
            Wt[k] -= lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v2[k] / (1 - 0.999 ** it)) + 1e-8)
        if it % 100 == 0 or it == steps:
            print(it, round(loss, 4), c.evaluate(Wt, test, np.random.default_rng(3)), f'{time.time() - t0:.0f}s', flush=True)
            np.savez(f'count_w_{tag}.npz', **Wt)
    c.MAXSIZE = 21
    print('final real positions   ', bybl(Wt, c.real(data, 'TEST', np.random.default_rng(2))[:3000]), flush=True)
    print('final balanced synthetic', bybl(Wt, c.synth_even(np.random.default_rng(99), 3000)), flush=True)
