"""Learning curves from count_multi.py's ckpt=1 checkpoints (ckpt/count_w_TAG_STEP.npz). Every checkpoint is scored on one fixed
balanced synthetic set (visible blues uniform 0-15: the finals' set, rng 99, first n trials) under the standard schedule and noise,
with the same decision rule and noise seed as the finals, so curves of different runs are directly comparable.
    python count_curves.py TAG [TAG ...] [n=1000]
Prints per run the accuracy (all / >4 blues / 11+ blues) at every checkpoint, then a comparison table: accuracy at fixed steps and
the first checkpoint at which the accuracy reached 80, 90 and 95% (a trailing mean of two checkpoints, so one lucky score does not
count). Merges the curves into count_curves.json."""
import os, sys, glob, json, re, numpy as np, torch, count as c, count_var as v, count_torch as ct
args = [a for a in sys.argv[1:] if '=' not in a]; kw = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a); n = int(kw.get('n', 1000))
AT = (250, 500, 1000, 1500, 2000, 3000, 4500, 6000)
c.SIG_REC, c.SIG_IN = 0.05, 0.02; v.PAUSE_MAX = 0; c.MAXSIZE = 21; c.NUMMAX = 15; c.PRE_W = 3.0
items = c.synth_even(np.random.default_rng(99), 3000)[:n]
sets = {}                                                                     # inputs per (make kind, NIN), built once

def score(path):
    P = dict(np.load(path)); c.N = P['J'].shape[0]; c.NIN = P['B'].shape[1]; pulse = 'pulse' in P
    key = (pulse, c.NIN)
    if key not in sets:
        c.make = v.make_gen(pulse=True, pos=False) if pulse else v.make0
        U, Y, M, meta = c.make(items, np.random.default_rng(0), interrupt=False)
        nb = np.array([m['st'].count('b') for m in meta]); sets[key] = (U, meta, nb)
    U, meta, nb = sets[key]
    Wt = ct.to_t({k: P[k] for k in ('J', 'B', 'c', 'W', 'cz')})
    alpha = torch.tensor(P['alpha'], dtype=torch.float32, device=ct.dev) if 'alpha' in P else None
    act = str(P['act']) if 'act' in P else 'tanh'; adapt = (float(P['adapt_beta']), float(P['adapt_g'])) if 'adapt_beta' in P else None
    gen = torch.Generator(device=ct.dev); gen.manual_seed(1)
    Z = ct.forward_chunks(Wt, U, gen, None, alpha, act, adapt); ok = np.zeros(len(meta), bool)
    for i, m in enumerate(meta):
        d = v.decide(Z[i, :m['L']], 'base')
        if d and d[0] >= m['det']: ok[i] = d[1] == m['lab']
    return float(ok.mean()), float(ok[nb > 4].mean()), float(ok[nb >= 11].mean())

def first_cross(steps, acc, thr):
    for j in range(1, len(acc)):
        if (acc[j - 1] + acc[j]) / 2 >= thr: return steps[j]
    return None

curves = json.load(open('count_curves.json')) if os.path.exists('count_curves.json') else {}
for tag in args:
    files = sorted(glob.glob(f'ckpt/count_w_{tag}_*.npz'), key=lambda f: int(re.search(r'_(\d+)\.npz$', f).group(1)))
    if not files: print(f'== {tag}: no checkpoints', flush=True); continue
    st, al, g4, b11 = [], [], [], []
    print(f'== {tag}  ({len(files)} checkpoints, {n} balanced trials each)', flush=True)
    for f in files:
        s = int(re.search(r'_(\d+)\.npz$', f).group(1)); a, b, d = score(f); st.append(s); al.append(a); g4.append(b); b11.append(d)
        print(f'  {s:5d}  all {a:.3f}  >4 {b:.3f}  11+ {d:.3f}', flush=True)
    curves[tag] = dict(steps=st, all=al, gt4=g4, b11=b11, n=n)
json.dump(curves, open('count_curves.json', 'w'))
print(f'\n== comparison: balanced accuracy at fixed steps; first checkpoint where it reached 80/90/95% (mean of two checkpoints)')
print('   tag             ' + ' '.join(f'{k:>5}' for k in AT) + '   80%   90%   95%')
for tag in args:
    if tag not in curves: continue
    s, a = curves[tag]['steps'], curves[tag]['all']
    at = [a[s.index(k)] if k in s else float('nan') for k in AT]
    print(f'   {tag:<15} ' + ' '.join(f'{x:5.2f}' for x in at) + '  ' + ' '.join(f'{str(first_cross(s, a, t) or "-"):>5}' for t in (0.8, 0.9, 0.95)))
