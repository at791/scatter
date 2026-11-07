"""Cross-sections of each number's trough for the game page.  python count_cross_export.py -> count_cross.json
Two axes per number n, both from the network itself (no task variables):
  e = the one-step push of the dark input (per-unit alpha * B[:, dark]), orthogonalised to the trough's mean tangent;
  h = the trace a hidden stone leaves 1 s later (no-stone control subtracted, averaged over count positions),
      orthogonalised to the mean tangent and to e.
Per count position k = 0..n: log speed of the map on the plane C_k + u h + v e (scan complete off and on), uint8 over
[-1.8, 0.9] like the other panels, and the flow there projected on (h, e). Per number: where the state sits on h after
m = 0..8 hidden stones (ticks), and c_j.h, c_j.e along the same spline as count_trough.json (SEG 8), so the page
measures the live state from the nearest trough point it already finds."""
import json, base64, numpy as np, count as c
Wt = dict(np.load('count_w_w_base.npz')); c.N = 300; c.NIN = 5; a = c.alpha_of(Wt)
land = json.load(open('count_land.json')); PL2 = np.array(land['P']); TRO = json.load(open('count_trough.json'))['nums']
PRE, FIX, GAP, PUL, AMP, SEG = 12, 12, 7, 3, 2.0, 8
U0, U1, V0, V1, NU, NV = -2.0, 10.0, -2.5, 6.5, 36, 28
def step(X, n, sc=0):
    u0 = np.zeros(5); u0[3] = n / 10; u0[4] = sc; return (1 - a) * X + a * (np.tanh(X) @ Wt['J'].T + u0 @ Wt['B'].T + Wt['c'])
def run(x, n, U):
    for u in U: x = (1 - a) * x + a * (np.tanh(x) @ Wt['J'].T + u @ Wt['B'].T + Wt['c'])
    return x
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
def stones(n, kinds, tail):                       # inputs for a run of stones 19 steps apart, then `tail` silent steps
    U = np.zeros((len(kinds) * (FIX + GAP) + tail, 5)); U[:, 3] = n / 10
    for j, ch in enumerate(kinds): U[j * (FIX + GAP):j * (FIX + GAP) + PUL, ch] = AMP
    return U
b8 = lambda A: base64.b64encode(np.asarray(A, np.uint8).tobytes()).decode()
FQ = 60                                                                              # flow packed as int8: component * FQ (range +-2.1)
i8 = lambda A: base64.b64encode(np.clip(np.round(np.asarray(A) * FQ), -127, 127).astype(np.int8).tobytes()).decode()
uu, vv = np.linspace(U0, U1, NU), np.linspace(V0, V1, NV); fu, fv = np.linspace(U0, U1, 11)[1:-1], np.linspace(V0, V1, 9)[1:-1]
out = dict(ext=[U0, U1, V0, V1], NU=NU, NV=NV, seg=SEG, fq=FQ, nums={}); Pd = a * Wt['B'][:, 1]
for n in range(1, 16):
    C = count_states(n); cv = catmull(C, SEG); tan = np.gradient(cv, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    tm = tan.mean(0); tm /= np.linalg.norm(tm); d = np.array(TRO[str(n)]['d'])
    assert np.allclose(np.round((cv - np.array(land['mu'])) @ PL2.T, 3)[:, 0], TRO[str(n)]['cx'], atol=2e-3)   # same spline as the trough sheet
    e = Pd - (Pd @ tm) * tm; e /= np.linalg.norm(e)
    hs = []
    for k in range(n):
        D = run(C[k], n, stones(n, [2], 31)) - run(C[k], n, stones(n, [], 50)); hs.append(D / np.linalg.norm(D))   # hidden trace 1 s after the stone
    h = np.mean(hs, 0); h -= (h @ tm) * tm; h -= (h @ e) * e; h /= np.linalg.norm(h)
    ticks = []                                                                       # where the state sits on h after m hidden stones (1 s after the last), mean over count positions
    for m in range(9):
        ks = range(0, n, max(1, n // 4)); ticks.append(float(np.mean([(run(C[k], n, stones(n, [2] * m, 50)) - run(C[k], n, stones(n, [], m * (FIX + GAP) + 50))) @ h for k in ks])))
    rec = dict(e=np.round(e, 5).tolist(), h=np.round(h, 5).tolist(), ce=np.round(cv @ e, 4).tolist(), ch=np.round(cv @ h, 4).tolist(), ticks=np.round(ticks, 3).tolist(),
               q0=[], q1=[], f0=[], f1=[])
    for k in range(n + 1):
        G = C[k][None, None, :] + uu[:, None, None] * h + vv[None, :, None] * e          # (NU, NV, 300)
        Fg = C[k][None, None, :] + fu[:, None, None] * h + fv[None, :, None] * e
        for sc in (0, 1):
            F = step(G.reshape(-1, 300), n, sc) - G.reshape(-1, 300); q = np.log10((F ** 2).sum(1) + 1e-12).reshape(NU, NV)
            rec['q%d' % sc].append(b8(np.clip(np.round((q.T[::-1] + 1.8) / 2.7 * 255), 0, 255)))   # rows top (v high) to bottom, columns u
            F2 = step(Fg.reshape(-1, 300), n, sc) - Fg.reshape(-1, 300)
            rec['f%d' % sc].append(i8(np.c_[F2 @ h, F2 @ e].ravel()))                    # (u outer, v inner), pairs (along h, along e)
    out['nums'][str(n)] = rec
    lt = [abs(tan[j] @ h) for j in range(len(cv))]; lte = [abs(tan[j] @ e) for j in range(len(cv))]
    print('number %2d: |P_land h| %.2f |P_land e| %.2f | cos(h, local tangent) max %.2f, (e) max %.2f | cos(h,d) %.2f cos(e,d) %.2f | hidden ticks %s'
          % (n, np.linalg.norm(PL2 @ h), np.linalg.norm(PL2 @ e), max(lt), max(lte), h @ d, e @ d, np.round(ticks, 2)))
out['fu'] = np.round(fu, 3).tolist(); out['fv'] = np.round(fv, 3).tolist()
json.dump(out, open('count_cross.json', 'w'), separators=(',', ':')); print('KB', len(json.dumps(out, separators=(',', ':'))) // 1024)
