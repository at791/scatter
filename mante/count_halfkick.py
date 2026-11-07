"""Line or wells? The half-kick test.  python count_halfkick.py CKPT.npz [...]
Noise off, number 8 (and 2), two blues 320 ms apart, then 1 s of silence: take the state after one blue (x1) and after two (x2),
d = x2 - x1 is one count step in state space. Start the net at x1 + f d for f = 0.25, 0.5, 0.75, 1.0, 1.5 and let it run silently
(number input kept) for 10 s. Report the projection of (x(t) - x1(t)) on d, in units of |d|, at 0, 1, 2, 5, 10 s, and the
distance off the d axis. A line attractor keeps f where it was put (projection stays ~f); point attractors pull it to the nearest
integer (0 or 1), the discrete signature the eigenvalue test cannot see in heterogeneous-tau nets."""
import sys, numpy as np, count as c, count_var as v

def run_silent(Wt, x0, u, T):
    """Noise-free integration of count.forward's dynamics from state x0 (K,N) under the constant input u (NIN,)."""
    al = c.alpha_of(Wt) * np.ones(Wt['J'].shape[0]); xs = [x0]; x = x0.copy()
    for _ in range(T):
        x = x + al * (-x + c.phi(x, Wt) @ Wt['J'].T + Wt['B'] @ u + Wt['c']); xs.append(x)
    return np.stack(xs, 1)                                                   # (K, T+1, N)

def halfkick(Wt, num, pulse):
    T0 = 15 + 16 + v.PULSE_LEN + 50; U = np.zeros((3, T0, c.NIN), np.float32); U[:, :, 3] = num / 10
    for k in (1, 2):
        for j in range(k):
            if pulse: U[k, 15 + 16 * j:15 + 16 * j + v.PULSE_LEN, 0] = v.PULSE_AMP
            else: U[k, 15 + 16 * j:27 + 16 * j, 0] = 1
    _, (xs, _) = c.forward(Wt, U, None, False, True); X = np.stack(xs, 1)[:, 1:]   # (3, T0, N) states; row 0 has no blue
    x0, x1, x2 = X[0, -1], X[1, -1], X[2, -1]; d = x2 - x1; dn = np.linalg.norm(d); dh = d / dn; u = U[1, -1].copy(); u[:3] = 0
    fs = (0.25, 0.5, 0.75, 1.0, 1.5); starts = np.stack([x1 + f * d for f in fs] + [x1, x2])
    tr = run_silent(Wt, starts, u, 500); ref1 = tr[-2]; ref2 = tr[-1]
    out = {}
    for i, f in enumerate(fs):
        rel = tr[i] - ref1; proj = rel @ dh / dn; off = np.linalg.norm(rel - np.outer(proj * dn, dh), axis=1) / dn
        out[f] = dict(proj={f'{t * 20 / 1000:g}s': round(float(proj[t]), 2) for t in (0, 50, 100, 250, 500)}, off_axis_10s=round(float(off[500]), 2))
    drift = round(float(np.linalg.norm(ref2[500] - ref1[500]) / dn), 2)   # does the step itself survive 10 s of silence (1 = fully held)
    return out, drift

for path in sys.argv[1:]:
    Wt = dict(np.load(path)); c.N = Wt['J'].shape[0]; c.NIN = Wt['B'].shape[1]; pulse = 'pulse' in Wt
    counter = Wt['W'].shape[0] == 16 or 'succ' in Wt                          # a pretrained counter (count_pre.py, count_succ.py): never saw a number input,
    print('==', path, '(pretrained counter, number input off)' if counter else '')   # so its B[:, 3] is untrained; test it at number 0
    for num in ((0,) if counter else (8, 2)):
        out, drift = halfkick(Wt, num, pulse)
        print(f'  number {num}: one-step distance after 10 s silence, in step units: {drift}')
        for f, o in out.items(): print(f'    start at {f:>4} step: projection {o["proj"]}  off-axis at 10 s {o["off_axis_10s"]}')
    sys.stdout.flush()
