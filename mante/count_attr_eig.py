"""Slow-point stability with the per-unit time constants taken into account.  python count_attr_eig.py CKPT.npz [...]
count_attr.py linearises with -I + J dphi, which assumes one tau for every unit. The map actually iterated is
x <- x + alpha_i (-x + J phi(x) + B u + c)_i, whose Jacobian is diag(alpha) (-I + J dphi). A positive diagonal scaling can change
the sign of eigenvalues, so for heterogeneous-tau nets the earlier top eigenvalues can be wrong. Reported here per 100 ms (x 1/ALPHA),
the same units as before, for the same gap states (noise on, 60% through the scan, number input kept)."""
import sys, numpy as np, count as c, count_var as v
c.SIG_REC, c.SIG_IN = 0.05, 0.02
test = c.real('../rnn/out_data.json', 'TEST', np.random.default_rng(2))[:600]

def slowpoints(Wt):
    N = Wt['J'].shape[0]; al = c.alpha_of(Wt) * np.ones(N)
    F = lambda x, u: -x + Wt['J'] @ c.phi(x, Wt) + Wt['B'] @ u + Wt['c']
    jac = lambda x: al[:, None] * (-np.eye(N) + Wt['J'] * c.dphi(x, Wt)[None, :]) / c.ALPHA
    U, Y, M, meta = c.make(test[:300], np.random.default_rng(0), interrupt=False); _, (xs, _) = c.forward(Wt, U, np.random.default_rng(3), True, True)
    tops = []; found = 0; n = 0
    for i in range(0, 300, 3):
        m = meta[i]; t = int(m['done'] * 0.6); u = U[i, t].copy(); u[:3] = 0; x = xs[t + 1][i].copy(); lam = 1e-2; n += 1
        for _ in range(300):
            f = F(x, u); Jm = -np.eye(N) + Wt['J'] * c.dphi(x, Wt)[None, :]; q = 0.5 * f @ f
            dx = -np.linalg.solve(Jm.T @ Jm + lam * np.eye(N), Jm.T @ f); f2 = F(x + dx, u)
            if 0.5 * f2 @ f2 < q: x = x + dx; lam = max(lam / 3, 1e-6)
            else: lam *= 5
            if q < 1e-10: break
        if 0.5 * F(x, u) @ F(x, u) < 1e-6: found += 1; tops.append(np.sort(np.linalg.eigvals(jac(x)).real)[::-1][:3])
    tops = np.array(tops)
    return dict(found=f'{found}/{n}', top_eig_median=[round(float(np.median(tops[:, k])), 3) for k in range(3)] if len(tops) else None,
                share_top_below_m0p05=round(float(np.mean(tops[:, 0] < -0.05)), 2) if len(tops) else None,
                share_top_below_m0p2=round(float(np.mean(tops[:, 0] < -0.2)), 2) if len(tops) else None,
                share_top_above_0=round(float(np.mean(tops[:, 0] > 0)), 2) if len(tops) else None)

for path in sys.argv[1:]:
    Wt = dict(np.load(path)); PULSE = 'pulse' in Wt; POS = 'pos' in Wt; c.NIN = Wt['B'].shape[1]
    c.make = v.make_gen(pulse=PULSE, pos=POS) if (PULSE or POS) else v.make0; v.PAUSE_MAX = 0
    print(path, slowpoints(Wt), flush=True)
