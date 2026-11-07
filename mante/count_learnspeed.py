"""Do pretrained nets learn the game faster? Learning curves from the 50-step scores that count_multi.py logs (800 real game
positions, the same set in every stack, scored under each run's training condition). Compares transfers from a pretrained counter
with fresh runs of the identical recipe (300 units: stack 3 w_base seeds 1-3 and stack 5 p_base_s4; 100 units: stack 2 s_pre
against s_seed3 and stack 1 p_hold).
    python count_learnspeed.py [TAG ...] [upto=STEP]   (default: the groups below, written to count_learnspeed.txt and .json;
    with tags, just those runs, printed only; upto cuts every curve at that step, for runs still training)"""
import re, sys, json, numpy as np
UPTO = int(next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('upto=')), 10 ** 9))
TAGS = [a for a in sys.argv[1:] if '=' not in a]
GROUPS = {'300 units': {'fresh': ['w_base', 'w_base_s2', 'w_base_s3', 'p_base_s4'],
                        'pretrained': ['p_transfer', 'p_slowJ', 'p_frozen_half', 'p_frozen_75', 'p_frozen']},
          '100 units': {'fresh': ['s_seed3', 'p_hold'], 'pretrained': ['s_pre']}}
AT = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000)
LINE = re.compile(r"^(\d+) [\d.]+ \{'acc': ([\d.]+)")

def curve(tag):
    pts = [(int(m.group(1)), float(m.group(2))) for m in map(LINE.match, open(f'count_{tag}.log')) if m]
    seen = {}
    for s, a in pts: seen[s] = a                                          # a resumed run may log a step twice; keep the last
    s = np.array([k for k in sorted(seen) if k <= UPTO]); return s, np.array([seen[k] for k in s])

def smooth(a, w=3):
    """Trailing mean over the last w scores (150 steps): no look-ahead, so a crossing step is when it was first reached."""
    return np.array([a[max(0, i - w + 1):i + 1].mean() for i in range(len(a))])

def first_cross(s, a, thr):
    i = np.flatnonzero(smooth(a) >= thr); return int(s[i[0]]) if len(i) else None

out, js = [], {}
if TAGS: GROUPS = {f'selected runs (curves cut at step {UPTO})' if UPTO < 10 ** 9 else 'selected runs': {'run': TAGS}}
for g, parts in GROUPS.items():
    out.append(f'== {g}: accuracy on the 800 test positions (trailing mean of 3 scores), and the first step where it reaches 80/90/95%')
    out.append('   tag             ' + ' '.join(f'{k:>5}' for k in AT) + '   80%   90%   95%  final')
    for kind, tags in parts.items():
        for t in tags:
            s, a = curve(t); sm = smooth(a); js[t] = dict(steps=s.tolist(), acc=a.tolist(), kind=kind, group=g)
            at = [sm[np.searchsorted(s, k)] if k <= s[-1] else np.nan for k in AT]
            cr = [first_cross(s, a, th) for th in (0.8, 0.9, 0.95)]
            out.append(f'   {t:<15} ' + ' '.join(f'{x:5.2f}' for x in at) + '  ' + ' '.join(f'{(c if c is not None else "-"):>5}' for c in cr) + f'  {a[-1]:5.2f}   [{kind}]')
    out.append('')
if not TAGS: open('count_learnspeed.txt', 'w').write('\n'.join(out) + '\n'); json.dump(js, open('count_learnspeed.json', 'w'))
print('\n'.join(out))
