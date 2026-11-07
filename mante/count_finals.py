"""Final by-blue scoring for finished count_multi runs whose in-process finals did not run.  python count_finals.py CONFIG.json [TAG ...]
Reproduces count_multi.py's four final lines per run from its checkpoint count_w_TAG.npz: training condition (the run's generator
with pause ceiling 0 and the run's noise) and the standard condition (pulse generator, noise 0.05/0.02), on 3000 real test positions
and 3000 balanced synthetic circles. Appends the lines to count_TAG.log and prints them."""
import os; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, json, numpy as np, torch, count as c, count_var as v, count_torch as ct
cfgs = json.load(open(sys.argv[1])); only = set(sys.argv[2:]); data = '../rnn/out_data.json'
for cfg in cfgs:
    tag = cfg['tag']
    if only and tag not in only: continue
    Wt = dict(np.load(f'count_w_{tag}.npz')); c.N = Wt['J'].shape[0]; c.NIN = Wt['B'].shape[1]
    W = ct.to_t({k: Wt[k] for k in ('J', 'B', 'c', 'W', 'cz')}); act = str(Wt['act']) if 'act' in Wt else 'tanh'
    kw = dict(in_mask=None, alpha=torch.tensor(Wt['alpha'], dtype=torch.float32, device=ct.dev) if 'alpha' in Wt else None, act=act,
              adapt=(float(Wt['adapt_beta']), float(Wt['adapt_g'])) if 'adapt_beta' in Wt else None)
    parts = set(cfg.get('gen', 'std').split('+')); pulse = 'pulse' in parts; pos = 'pos' in parts
    log = open(f'count_{tag}.log', 'a')
    def say(s): print(f'[{tag}] {s}', flush=True); log.write(s + '\n'); log.flush()
    c.MAXSIZE = 21; c.PRE_W = 3.0; v.PAUSE_MAX = 0; c.SIG_REC, c.SIG_IN = cfg.get('sig', (0.05, 0.02))
    c.make = v.make_gen(pause=bool(parts & {'pause', 'hold'}), wide=bool(parts & {'fix', 'wide', 'hold'}), pulse=pulse, curric='pausecur' in parts, pos=pos) if cfg.get('gen', 'std') != 'std' else v.make0
    say('final (training condition) real positions    ' + ct.bybl(W, c.real(data, 'TEST', np.random.default_rng(2))[:3000], cfg.get('var', 'base'), **kw))
    say('final (training condition) balanced synthetic ' + ct.bybl(W, c.synth_even(np.random.default_rng(99), 3000), cfg.get('var', 'base'), **kw))
    c.make = v.make_gen(pulse=pulse, pos=pos) if (pulse or pos) else v.make0; c.SIG_REC, c.SIG_IN = 0.05, 0.02
    say('final (standard schedule+noise) real positions    ' + ct.bybl(W, c.real(data, 'TEST', np.random.default_rng(2))[:3000], cfg.get('var', 'base'), **kw))
    say('final (standard schedule+noise) balanced synthetic ' + ct.bybl(W, c.synth_even(np.random.default_rng(99), 3000), cfg.get('var', 'base'), **kw))
    log.close()
