"""Line or basins? Compact attractor measurements for a base-variant checkpoint.   python count_attr.py CKPT.npz [CKPT2.npz ...]
1. hold: accuracy on real positions with a pause of 0/2/5/10 s inserted mid-scan (noise on, standard level)
2. slow points during gaps: fraction converged, top Jacobian eigenvalue (median, and share clearly negative = stable point)
3. need code at number 8: distance from the no-blue state after k blues (2 s after the last), and the single-blue kick over 0..10 s
"""
import sys, numpy as np, count as c, count_var as v
c.SIG_REC, c.SIG_IN = 0.05, 0.02; N = c.N; PULSE = False                  # checkpoints trained on onset pulses carry 'pulse'
# checkpoints may carry per-unit 'alpha' and 'act' (see count.forward); only J/B/c/W/cz are weights
test = c.real('../rnn/out_data.json', 'TEST', np.random.default_rng(2))[:600]

def hold(Wt, gap):
    U2, Y2, M2, meta2 = c.make(test, np.random.default_rng(0), interrupt=False); B, T, _ = U2.shape; U3 = np.zeros((B, T + gap, c.NIN), np.float32)
    for i, m in enumerate(meta2):
        cut = int(np.random.default_rng(i).integers(5, max(6, m['done'] - 5)))
        U3[i, :cut] = U2[i, :cut]; U3[i, cut:cut + gap, 3] = U2[i, cut - 1, 3]; U3[i, cut + gap:] = U2[i, cut:]
    Z3, _ = c.forward(Wt, U3, np.random.default_rng(1), True); ok = early = 0
    for i, m in enumerate(meta2):
        z = Z3[i, :m['L'] + gap]; ab = np.where(z.max(1) > 0.5)[0]
        if not len(ab): continue
        if ab[0] < m['det']: early += 1
        elif int(z[ab[0]].argmax()) == m['lab']: ok += 1
    return ok / B, early / B

def slowpoints(Wt):
    F = lambda x, u: -x + Wt['J'] @ c.phi(x, Wt) + Wt['B'] @ u + Wt['c']
    jac = lambda x: -np.eye(N) + Wt['J'] * c.dphi(x, Wt)[None, :]
    U, Y, M, meta = c.make(test[:300], np.random.default_rng(0), interrupt=False); _, (xs, _) = c.forward(Wt, U, np.random.default_rng(3), True, True)
    tops = []; found = 0; n = 0
    for i in range(0, 300, 3):
        m = meta[i]; t = int(m['done'] * 0.6); u = U[i, t].copy(); u[:3] = 0; x = xs[t + 1][i].copy(); lam = 1e-2; n += 1
        for _ in range(300):
            f = F(x, u); Jm = jac(x); q = 0.5 * f @ f
            dx = -np.linalg.solve(Jm.T @ Jm + lam * np.eye(N), Jm.T @ f); f2 = F(x + dx, u)
            if 0.5 * f2 @ f2 < q: x = x + dx; lam = max(lam / 3, 1e-6)
            else: lam *= 5
            if q < 1e-10: break
        if 0.5 * F(x, u) @ F(x, u) < 1e-6: found += 1; tops.append(np.sort(np.linalg.eigvals(jac(x)).real)[::-1][:3])
    tops = np.array(tops)
    return dict(found=f'{found}/{n}', top_eig_median=[round(float(np.median(tops[:, k])), 3) for k in range(3)] if len(tops) else None,
                share_top_below_m0p05=round(float(np.mean(tops[:, 0] < -0.05)), 2) if len(tops) else None,
                share_top_below_m0p2=round(float(np.mean(tops[:, 0] < -0.2)), 2) if len(tops) else None)

def needcode(Wt, num=8):
    """Noise off. Blues at 16-step spacing (12 fixated, 4 gap); distances between population states (tanh) at matched times."""
    T = 15 + 16 * 8 + 500; K = 8
    U = np.zeros((K, T, c.NIN), np.float32); U[:, :, 3] = num / 10
    for k in range(K):
        for j in range(k):
            if PULSE: U[k, 15 + 16 * j:15 + 16 * j + v.PULSE_LEN, 0] = v.PULSE_AMP
            else: U[k, 15 + 16 * j:27 + 16 * j, 0] = 1
    _, (xs, _) = c.forward(Wt, U, None, False, True); R = c.phi(np.stack(xs, 1)[:, 1:], Wt)
    t_end = 15 + 16 * 7 + 100                                                   # 2 s after the last possible blue
    d_by_k = [round(float(np.linalg.norm(R[k, t_end] - R[0, t_end])), 2) for k in range(1, K)]
    steps = [round(float(np.linalg.norm(R[k, t_end] - R[k - 1, t_end])), 2) for k in range(1, K)]
    t0 = 27; kick = {f'{(t - t0) * 20 / 1000:g}s': round(float(np.linalg.norm(R[1, t] - R[0, t])), 2) for t in (t0, t0 + 25, t0 + 50, t0 + 100, t0 + 250, t0 + 500)}
    return dict(dist_from_zero_blues_by_k=d_by_k, step_k_minus_1_to_k=steps, single_blue_kick_vs_time=kick)

for path in sys.argv[1:]:
    Wt = dict(np.load(path)); N = Wt['J'].shape[0]; PULSE = 'pulse' in Wt; POS = 'pos' in Wt; c.NIN = Wt['B'].shape[1]
    c.make = v.make_gen(pulse=PULSE, pos=POS) if (PULSE or POS) else v.make0; v.PAUSE_MAX = 0
    print('==', path, f"act {c.act_of(Wt)}, alpha {'per-unit' if 'alpha' in Wt else c.ALPHA}, input {'pulse' if PULSE else 'sustained'}"
          f"{' + position' if POS else ''}, adapt {c.adapt_of(Wt)}")
    print('  hold (acc, early) by pause:', {f'{g * 20 / 1000:g}s': tuple(round(v, 3) for v in hold(Wt, g)) for g in (0, 100, 250, 500)})
    print('  slow points:', slowpoints(Wt))
    nc = needcode(Wt); print('  number 8, state distance from 0 blues after k=1..7 blues:', nc['dist_from_zero_blues_by_k'])
    print('  number 8, step from k-1 to k:', nc['step_k_minus_1_to_k']); print('  number 8, single-blue kick vs time:', nc['single_blue_kick_vs_time'])
    nc2 = needcode(Wt, 2); print('  number 2, single-blue kick vs time:', nc2['single_blue_kick_vs_time'], flush=True)
