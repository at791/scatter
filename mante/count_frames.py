"""Frames for the counting animation of the serial network: fixation sequence, three outputs, and a population plane
built from the network's own unit groups (plane fit of each unit's need x hidden map on real positions)."""
import sys, json, numpy as np
import count as c
Wt = dict(np.load(sys.argv[2])); N = c.N
test = c.real(sys.argv[1], 'TEST', np.random.default_rng(2))
rng = np.random.default_rng(5)
# --- unit groups from real positions at end of scan (no noise) ---
sub = test[:3000]; U, Y, M, meta = c.make(sub, np.random.default_rng(0), interrupt=False); Z, (xs, _) = c.forward(Wt, U, rng, False, True)
R = np.tanh(np.stack(xs, 1))[:, 1:]
E = np.array([R[i, meta[i]['done'] + 2] for i in range(len(meta))])
need = np.array([m['num'] - m['st'].count('b') for m in meta]); hid = np.array([m['st'].count('h') for m in meta])
G = np.full((6, 9, N), np.nan); cnt = np.zeros((6, 9), int)
for v in range(6):
    for h in range(1, 9):
        m = (need == v) & (hid == h)
        if m.sum() >= 15: G[v, h] = E[m].mean(0); cnt[v, h] = m.sum()
rows = [(v, h, cnt[v, h]) for v in range(6) for h in range(1, 9) if cnt[v, h] >= 15]
V = np.array([r[0] for r in rows], float); H = np.array([r[1] for r in rows], float); Wt_ = np.sqrt([r[2] for r in rows])
X = np.c_[np.ones(len(V)), V, H] * Wt_[:, None]; ang = {}; gain = {}
for u in range(N):
    y = np.array([G[v, h, u] for v, h, _ in rows]); rg = np.nanmax(y) - np.nanmin(y)
    if rg < 0.4: continue
    coef, *_ = np.linalg.lstsq(X, y * Wt_, rcond=None); b, cc = coef[1], coef[2]
    a = np.degrees(np.arctan2(cc, b)); a = a - 180 if a > 90 else a + 180 if a <= -90 else a
    ang[u] = float(a); gain[u] = float(np.hypot(b, cc))
need_u = [u for u, a in ang.items() if -20 < a <= 20]; slack_u = [u for u, a in ang.items() if -60 <= a <= -35]
def sgn(u, key):
    v = [np.nanmean(G[n, :, u]) for n in range(6)] if key == 'need' else [np.nanmean([G[n, h, u] for n in range(6) for h in range(1, 9) if n - h == s]) for s in range(-5, 1)]
    v = np.array(v); ok = ~np.isnan(v); x = np.arange(len(v))[ok]; return float(np.sign(np.polyfit(x, v[ok], 1)[0])) if ok.sum() >= 3 else 1.0
sn = np.array([sgn(u, 'need') for u in need_u]); ss = np.array([sgn(u, 'slack') for u in slack_u])
proj = lambda r: (np.mean(r[..., need_u] * sn, axis=-1) if need_u else np.zeros(r.shape[:-1]), np.mean(r[..., slack_u] * ss, axis=-1) if slack_u else np.zeros(r.shape[:-1]))
xn, ys = proj(E); slack = need - hid
ref = dict(need={int(v): round(float(xn[need == v].mean()), 3) for v in range(0, 6) if (need == v).sum() > 10},
           slack={int(v): round(float(ys[slack == v].mean()), 3) for v in range(-4, 1) if (slack == v).sum() > 10})
angles = sorted([[u, round(a, 1), round(gain[u], 3)] for u, a in ang.items()], key=lambda r: r[1])
# --- example trials (one per verdict), no noise, no interruptions ---
pick = []
for lab in (1, 0, 2):
    for num, st in test:
        if c.label(num, st.count('b'), st.count('h')) == lab and 7 <= len(st) <= 9 and st.count('b') >= 2 and st.count('h') >= 2 and st.count('d') >= 2: pick.append((num, st)); break
U1, Y1, M1, meta1 = c.make(pick, np.random.default_rng(11), interrupt=False); Z1, (xs1, _) = c.forward(Wt, U1, rng, False, True); R1 = np.tanh(np.stack(xs1, 1))[:, 1:]
ex = []
for i, m in enumerate(meta1):
    L = m['L']; fix = [int(np.argmax(U1[i, t, :3])) if U1[i, t, :3].max() > 0.5 else -1 for t in range(L)]
    # which stone index is fixated at each step
    idx = []; j = -1
    for t in range(L):
        if fix[t] >= 0 and (t == 0 or fix[t - 1] < 0): j += 1
        idx.append(j if fix[t] >= 0 else -1)
    a, b = proj(R1[i, :L]); ex.append(dict(num=m['num'], st=m['st'], lab=m['lab'], det=m['det'], done=m['done'], L=L, fix=idx,
                                            z=np.round(Z1[i, :L], 2).tolist(), pn=np.round(a, 3).tolist(), ps=np.round(b, 3).tolist()))
out = dict(need_u=need_u, slack_u=slack_u, sn=sn.tolist(), ss=ss.tolist(), angles=angles, ref=ref, ex=ex)
json.dump(out, open('count_frames.json', 'w'), separators=(',', ':'))
print('need units', len(need_u), 'slack units', len(slack_u), 'strong', len(ang)); print('ref', ref); print([(e['num'], e['st'], e['lab'], e['det'], e['done'], e['L']) for e in ex])
