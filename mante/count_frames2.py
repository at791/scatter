"""Unit groups and reference ticks for the artifact's population plane, for a pulse-input / heterogeneous-tau checkpoint.
python count_frames2.py ../rnn/out_data.json count_w_w_base.npz count_frames_w.json
Same analysis as count_frames.py (plane fit of each unit's end-of-scan activity on real positions in need x hidden; need-type
within 20 degrees of the need axis, slack-type 35-60 degrees towards need - hidden), with the pulse generator and the
checkpoint's own per-unit alpha (count.forward reads it), no noise."""
import sys, json, numpy as np, count as c, count_var as v
Wt = dict(np.load(sys.argv[2])); N = Wt['J'].shape[0]; c.N = N; c.NIN = Wt['B'].shape[1]
c.make = v.make_gen(pulse='pulse' in Wt); v.PAUSE_MAX = 0; c.MAXSIZE = 21
test = c.real(sys.argv[1], 'TEST', np.random.default_rng(2)); rng = np.random.default_rng(5)
sub = test[:3000]; U, Y, M, meta = c.make(sub, np.random.default_rng(0), interrupt=False); Z, (xs, _) = c.forward(Wt, U, rng, False, True)
R = np.tanh(np.stack(xs, 1))[:, 1:]
E = np.array([R[i, meta[i]['done'] + 2] for i in range(len(meta))])
need = np.array([m['num'] - m['st'].count('b') for m in meta]); hid = np.array([m['st'].count('h') for m in meta])
G = np.full((6, 9, N), np.nan); cnt = np.zeros((6, 9), int)
for vv in range(6):
    for h in range(1, 9):
        m = (need == vv) & (hid == h)
        if m.sum() >= 15: G[vv, h] = E[m].mean(0); cnt[vv, h] = m.sum()
rows = [(vv, h, cnt[vv, h]) for vv in range(6) for h in range(1, 9) if cnt[vv, h] >= 15]
V = np.array([r[0] for r in rows], float); H = np.array([r[1] for r in rows], float); Wt_ = np.sqrt([r[2] for r in rows])
X = np.c_[np.ones(len(V)), V, H] * Wt_[:, None]; ang = {}; gain = {}
for u in range(N):
    y = np.array([G[vv, h, u] for vv, h, _ in rows]); rg = np.nanmax(y) - np.nanmin(y)
    if rg < 0.4: continue
    coef, *_ = np.linalg.lstsq(X, y * Wt_, rcond=None); b, cc = coef[1], coef[2]
    a = np.degrees(np.arctan2(cc, b)); a = a - 180 if a > 90 else a + 180 if a <= -90 else a
    ang[u] = float(a); gain[u] = float(np.hypot(b, cc))
need_u = [u for u, a in ang.items() if -20 < a <= 20]; slack_u = [u for u, a in ang.items() if -60 <= a <= -35]
def sgn(u, key):
    vals = [np.nanmean(G[n, :, u]) for n in range(6)] if key == 'need' else [np.nanmean([G[n, h, u] for n in range(6) for h in range(1, 9) if n - h == s]) for s in range(-5, 1)]
    vals = np.array(vals); ok = ~np.isnan(vals); x = np.arange(len(vals))[ok]; return float(np.sign(np.polyfit(x, vals[ok], 1)[0])) if ok.sum() >= 3 else 1.0
sn = np.array([sgn(u, 'need') for u in need_u]); ss = np.array([sgn(u, 'slack') for u in slack_u])
proj = lambda r: (np.mean(r[..., need_u] * sn, axis=-1) if need_u else np.zeros(r.shape[:-1]), np.mean(r[..., slack_u] * ss, axis=-1) if slack_u else np.zeros(r.shape[:-1]))
xn, ys = proj(E); slack = need - hid
ref = dict(need={int(vv): round(float(xn[need == vv].mean()), 3) for vv in range(0, 6) if (need == vv).sum() > 10},
           slack={int(vv): round(float(ys[slack == vv].mean()), 3) for vv in range(-4, 1) if (slack == vv).sum() > 10})
angles = sorted([[u, round(a, 1), round(gain[u], 3)] for u, a in ang.items()], key=lambda r: r[1])
out = dict(N=N, need_u=need_u, slack_u=slack_u, sn=sn.tolist(), ss=ss.tolist(), angles=angles, ref=ref)
json.dump(out, open(sys.argv[3], 'w'), separators=(',', ':'))
print('N', N, 'need units', len(need_u), 'slack units', len(slack_u), 'strong', len(ang)); print('ref', ref)
print('projection ranges: need axis', round(float(xn.min()), 2), round(float(xn.max()), 2), 'slack axis', round(float(ys.min()), 2), round(float(ys.max()), 2))
