"""Train several configurations at once in one GPU process: each run keeps its own weights, generator, noise, penalty, loss and
learning-rate schedule, but all runs step together with batched matmuls, so one launch sequence serves the whole stack.
    python count_multi.py CONFIG.json [micro=256]
micro: trials per forward/backward chunk (gradients accumulate; same gradient as the whole batch, less memory, more launches).
CONFIG is a list of runs, each a dict with: tag (required), and optionally
    var: base | digit          (default base; 'two'/'commit'/'big' are not stackable here)
    init: fresh | PATH.npz     (default count_w_skewed.npz; fresh uses the staged recipe, otherwise lr decays 10x)
    steps (default 1000, must be equal across the stack), lr (5e-4 resumed / 1e-3 fresh), batch (512, equal across the stack),
    gain (1.0, fresh only), loss: std | norm (std), l2r (1e-4), sig: [rec, in] ([0.05, 0.02]), gen: std | pause | fix | hold (std),
    seed (per-run data seed; default 11 resumed, 1 fresh)
    tau: std | hetero (+ tau_frac, tau_range [lo, hi] ms; default a third of the units at 1-2 s), act: tanh | softplus,
    hold (gap speed penalty), adapt {tau, g}, sym, pausecur {max, frac}, gapnoise (recurrent noise x this in gaps, 1 = off),
    selfexc (added to the diagonal of J at init), transfer (init is a count_pre.py checkpoint: J, B, c, alpha are taken from it,
    the readout is fresh and the staged fresh schedule is used), staged (default true; false drops the staged recipe's size and
    silence-weight curriculum from a fresh or transfer run: full-size circles and silence weight 3 from the start, lr decaying 10x)
Writes count_w_TAG.npz every 50 steps and count_TAG.log per run, with the same final by-blue-count lines as count_torch.py.
Speed (count_speed.py): the allocator setting below stops fragmentation (reserved memory looked like 8 GB when 2.8 GB was in use);
each micro-chunk runs only to its own longest trial (trials are sorted by length); batches are generated one step ahead in a worker process.
Resume: the full training state (weights, Adam state, VQ codebooks, noise generator, step) is written to resume_CONFIG.pt every 50 steps
and at a stop; a rerun of the same config continues from it (batches are seeded per step, so a resumed run follows the same path as
an uninterrupted one) and appends to the count_TAG.log files. After the finals the file is marked finished and a rerun exits at once.
noresume ignores the file. Stop: until=HH:MM or until=EPOCH_SECONDS (default: the MANTE_DEADLINE environment variable, set by
run_stack.ps1) saves the state and exits with code 3 at the first step past that time; stopafter=K does the same after K steps (tests).
Out of memory in a training step halves micro (down to 16) and redoes the step; in the 50-step scores it skips those scores.
ckpt=1 also keeps learning-curve checkpoints ckpt/count_w_TAG_STEP.npz (weights in float32): the start, every 50 steps through
step 1000, then every 250."""
import os; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, json, time, numpy as np, torch, count as c, count_var as v, count_torch as ct
from concurrent.futures import ProcessPoolExecutor
dev = ct.dev

def hetero_alpha(N, seed=123, frac=1 / 3, lo=1000.0, hi=2000.0):
    """Per-unit dt/tau: a fraction of the units get tau log-uniform in [lo, hi] ms, the rest keep the standard 100 ms."""
    g = np.random.default_rng(seed); al = np.full(N, c.ALPHA); idx = g.choice(N, int(round(N * frac)), replace=False)
    al[idx] = 20.0 / np.exp(g.uniform(np.log(lo), np.log(hi), len(idx)))
    return al

def build(cfg):
    """Per-run state: weights (torch tensors), generator function, dims, noise, schedule."""
    r = dict(tag=cfg['tag'], var=cfg.get('var', 'base'), fresh=cfg.get('init', '') == 'fresh', steps=int(cfg.get('steps', 1000)),
             batch=int(cfg.get('batch', 512)), gain=float(cfg.get('gain', 1.0)), loss=cfg.get('loss', 'std'), l2r=float(cfg.get('l2r', c.L2R)),
             sig=tuple(cfg.get('sig', (c.SIG_REC, c.SIG_IN))), gen=cfg.get('gen', 'std'))
    r['lr0'] = float(cfg.get('lr', 1e-3 if r['fresh'] else 5e-4)); r['rng'] = np.random.default_rng(int(cfg.get('seed', 1 if r['fresh'] else 11)))
    parts = set(r['gen'].split('+'))                            # gen: std | pause | fix | hold (pause+fix) | pulse | pausecur | pos, joined with '+'
    r['pulse'] = 'pulse' in parts; r['pause'] = bool(parts & {'pause', 'hold'}); r['wide'] = bool(parts & {'fix', 'wide', 'hold'})
    r['curric'] = 'pausecur' in parts; r['pos'] = 'pos' in parts
    pc = cfg.get('pausecur', {}); r['pc_max'] = int(pc.get('max', 500)); r['pc_frac'] = float(pc.get('frac', 0.6))   # pause ceiling (steps), reached at this fraction of training
    ad = cfg.get('adapt'); r['adapt'] = (20.0 / float(ad.get('tau', 500)), float(ad.get('g', 0.5))) if ad else None    # (beta = dt/tau_a, strength g)
    r['sym'] = bool(cfg.get('sym', False))                      # symmetric recurrent weights: the forward uses (J + J^T) / 2
    r['hold'] = float(cfg.get('hold', 0.0))                     # speed penalty during gaps (no stone input): hold * mean squared deterministic speed per unit
    r['gapnoise'] = float(cfg.get('gapnoise', 1.0))             # recurrent noise multiplied by this at gap steps (same mask as the hold penalty)
    r['transfer'] = bool(cfg.get('transfer', False))            # init from a pretrained counter (count_pre.py): recurrent weights kept, readout fresh
    if r['transfer']: r['fresh'] = True
    r['staged'] = bool(cfg.get('staged', True))                 # False: a fresh/transfer run without the staged recipe (no size or silence-weight curriculum; lr decays 10x)
    init = None if r['fresh'] else cfg.get('init', None)
    c.N = int(cfg.get('N', 100))                                # width (fresh runs only; a checkpoint brings its own)
    r['evenfrac'] = float(cfg.get('evenfrac', 0.0))             # share of each batch drawn from synth_even (visible blues uniform 0..15) instead of synth
    Wt = v.setup(r['var'], r['fresh'], r['rng'], init)          # sets c.N / c.NIN / c.make for this var; we record them per run
    if r['fresh'] and r['gain'] != 1.0: Wt['J'] = Wt['J'] * r['gain']
    if cfg.get('selfexc'): Wt['J'] = Wt['J'] + float(cfg['selfexc']) * np.eye(c.N)   # self-excitation: units individually bistable when > 1
    r['bistable'] = float(cfg.get('bistable', 0.0))             # FIXED self-coupling: J_ii = s, never trained (J_of masks the learned diagonal); every unit bistable when s > 1
    r['eye'] = torch.eye(c.N, device=dev) if r['bistable'] else None
    if r['transfer']:
        P = dict(np.load(cfg['init'])); Wt['J'] = P['J']; Wt['c'] = P['c']; Wt['B'][:, :P['B'].shape[1]] = P['B'][:, :Wt['B'].shape[1]]
    if r['sym']: Wt['J'] = (Wt['J'] + Wt['J'].T) / 2
    if r['pos']: Wt['B'] = np.concatenate([Wt['B'], r['rng'].normal(0, 0.5, (c.N, 2))], 1)   # two extra input columns for cos/sin of the fixated angle
    r['rank'] = int(cfg.get('rank', 0))                         # low-rank recurrent weights J = Ju Jv^T (Mastrogiuseppe & Ostojic 2018), trained as the two factors
    if r['rank']:
        assert not r['sym'] and not r['transfer'], 'rank: fresh, non-symmetric runs only'
        s = (1.0 / c.N) ** 0.25 / r['rank'] ** 0.25                # entries of Ju Jv^T then have variance 1/N, the scale of the fresh init
        Wt['Ju'] = r['rng'].normal(0, s, (c.N, r['rank'])); Wt['Jv'] = r['rng'].normal(0, s, (c.N, r['rank'])); del Wt['J']
    cc = cfg.get('countcur'); r['countcur'] = [int(x) for x in cc['stages']] if cc else None   # count curriculum: blue/number ceiling steps through these over the first half
    r['freeze'] = {k: float(fr) for k, fr in cfg.get('freeze', {}).items()}   # {'J': 0.5}: J gets no updates for the first half of training (1.0 = never updated)
    r['aux_w'] = float(cfg.get('aux', {}).get('w', 0.0))                 # auxiliary 16-way running-count readout (Wa, cza) trained with this weight; never a game output
    ga = cfg.get('gain_anneal'); r['gain_anneal'] = (float(ga['to']), float(ga.get('frac', 0.5))) if ga else None   # tanh slope 1 -> to over the first frac of training
    vq = cfg.get('vq'); r['vq'] = dict(w=float(vq['w']), K=int(vq.get('K', 16)), rho=float(vq.get('rho', 0.05)), E=None) if vq else None   # quantised-state bottleneck at gap steps
    r['gam_now'] = 1.0
    if r['aux_w']:
        pre16 = r['transfer'] and P['W'].shape[0] == 16                  # a pretrained counter's count readout seeds the auxiliary readout
        Wt['Wa'] = P['W'] if pre16 else np.zeros((16, c.N)); Wt['cza'] = P['cz'] if pre16 else np.zeros(16)
    r['N'], r['NIN'] = c.N, Wt['B'].shape[1]
    r['make'] = v.make_gen(pause=r['pause'], wide=r['wide'], pulse=r['pulse'], curric=r['curric'], pos=r['pos']) if r['gen'] != 'std' else c.make
    r['in_mask'] = None
    if r['var'] == 'digit': r['in_mask'] = torch.zeros(c.NIN, device=dev); r['in_mask'][:5] = 1
    r['act'] = cfg.get('act', 'tanh')                           # 'tanh' or 'softplus' (unbounded rates)
    r['alpha'] = None                                           # per-unit dt/tau; tau='hetero' slows a third of the units to 1-2 s
    if cfg.get('tau', 'std') == 'hetero':
        lo, hi = cfg.get('tau_range', (1000.0, 2000.0))
        r['alpha'] = torch.tensor(hetero_alpha(c.N, int(cfg.get('tau_seed', 123)), float(cfg.get('tau_frac', 1 / 3)), float(lo), float(hi)), dtype=torch.float32, device=dev)
    if r['transfer'] and 'alpha' in P: r['alpha'] = torch.tensor(P['alpha'], dtype=torch.float32, device=dev)
    r['W'] = ct.to_t(Wt)
    r['lr_scale'] = {k: float(s) for k, s in cfg.get('lr_scale', {}).items()}   # {'J': 0.1}: J learns 10x slower than everything else
    r['opt'] = torch.optim.Adam([{'params': [w], 'lr_scale': r['lr_scale'].get(k, 1.0)} for k, w in r['W'].items()], lr=r['lr0'])
    r['spec'] = dict(tag=r['tag'], seed=int(cfg.get('seed', 1 if r['fresh'] else 11)) + 1000, var=r['var'], N=c.N, NIN=r['NIN'], batch=r['batch'],
                     evenfrac=r['evenfrac'], gen=dict(pause=r['pause'], wide=r['wide'], pulse=r['pulse'], curric=r['curric'], pos=r['pos']) if r['gen'] != 'std' else None)
    c.N, c.NIN, c.make = 100, 5, v.make0                        # restore module globals
    return r

def activate(r):
    """Point the count module at this run's dims, generator and noise (for batch making and evaluation)."""
    c.N, c.NIN, c.make = r['N'], r['NIN'], r['make']; c.SIG_REC, c.SIG_IN = r['sig']

def forward_stack(Ws, U, sr, si, in_mask, gen, N, alpha, soft, adapt=None, nscale=None, gam=None):
    """Ws: dict of stacked weights (R,...). U: (R,B,T,NINmax). sr, si: (R,1,1,1) noise scales. alpha: (R,1,N) per-run, per-unit dt/tau.
    soft: (R,1,1) bool, runs whose units are softplus rather than tanh. Returns Z (R,B,T,NOUT), R (R,B,T,N) and S (R,B,T), the
    squared deterministic speed ||x_t+1 - x_t||^2 before the noise is added (Haviv et al. 2019's speed, used by the hold penalty).
    adapt: None or (beta (R,1,1), g (R,1,1)); runs without adaptation have g = 0. nscale: None or (R,B,T,1) per-step recurrent noise factor."""
    Rn, B, T, _ = U.shape
    if si is not None: U = U + torch.randn(U.shape, device=dev, generator=gen) * si * in_mask
    pre = torch.baddbmm(Ws['c'][:, None, :], U.reshape(Rn, B * T, -1), Ws['B'].transpose(1, 2)).reshape(Rn, B, T, N)
    xn = torch.randn(Rn, B, T, N, device=dev, generator=gen) * (sr * alpha[:, None]) if sr is not None else None
    if xn is not None and nscale is not None: xn = xn * nscale
    if xn is not None and gam is not None: xn = xn / gam.view(-1, 1, 1, 1)   # gain runs: recurrent noise fixed in x' = gam x units, as for a standard net
    anysoft = bool(soft.any()); th = torch.tanh if gam is None else (lambda x: torch.tanh(gam * x))   # gam (R,1,1): annealed tanh slope
    f = (lambda x: torch.where(soft, torch.nn.functional.softplus(x), th(x))) if anysoft else th
    JT = Ws['J'].transpose(1, 2); x = torch.zeros(Rn, B, N, device=dev); r = f(x); rs = []; xds = []
    av = torch.zeros(Rn, B, N, device=dev) if adapt is not None else None
    pres = pre.unbind(2); xns = xn.unbind(2) if xn is not None else None   # unbind: one stack kernel in backward, not T dense zero-fills
    for t in range(T):                                                      # four kernels per step: baddbmm, lerp, add noise, nonlinearity
        drive = torch.baddbmm(pres[t], r, JT)
        if av is not None: drive = drive - adapt[1] * av
        xd = torch.lerp(x, drive, alpha); xds.append(xd)
        x = xd + xns[t] if xns is not None else xd
        r = f(x); rs.append(r)
        if av is not None: av = torch.lerp(av, r, adapt[0])
    R = torch.stack(rs, 2); XD = torch.stack(xds, 2)
    Xprev = torch.cat([torch.zeros_like(XD[:, :, :1]), XD[:, :, :-1] + (xn[:, :, :-1] if xn is not None else 0)], 2)   # x_t = xd_{t-1} + noise_{t-1}, x_0 = 0
    S = ((XD - Xprev) ** 2).sum(-1)                                         # deterministic speed, computed once after the loop (was three kernels per step)
    return torch.matmul(R, Ws['W'].transpose(1, 2)[:, None]) + Ws['cz'][:, None, None], R, S

def losses(runs, Z, R, Y, M, metas, scols):
    """Per-run loss (R,): std or norm as each run asks, plus its own activity penalty and the shared weight decay."""
    out = []
    for i, r in enumerate(runs):
        if r['loss'] == 'norm': base = ct.loss_norm(Z[i], Y[i], metas[i], scols[i])
        else: base = (M[i][:, :, None] * (Z[i] - Y[i]) ** 2).sum() / (M[i].sum() * c.NOUT)
        out.append(base + r['l2r'] * (R[i] ** 2).mean() + c.L2W * (J_of(r) ** 2).sum())
    return torch.stack(out)

SETTLE = 8                                                                  # steps (160 ms) after any stone input before the hold penalty applies

def gap_mask(U, metas):
    """(R,B,T) float: 1 at steps with no stone input now or in the previous SETTLE steps (so transients after a stone are not
    charged), before the trial ends. The hold penalty charges the deterministic speed at these steps."""
    stone = (U[:, :, :, :3].abs().sum(-1) > 0).float(); Rn, B, T = stone.shape
    recent = torch.nn.functional.max_pool1d(torch.nn.functional.pad(stone.reshape(Rn * B, 1, T), (SETTLE, 0)), SETTLE + 1, 1).reshape(Rn, B, T)
    G = 1 - recent
    for i, meta in enumerate(metas):
        for b, m in enumerate(meta): G[i, b, m['L']:] = 0
    return G

def valid_mask(metas, B, T):
    """(R,B,T) float: 1 at steps inside each trial (t < L). The activity penalty averages over these steps only, so it does not
    depend on how a batch is padded or chunked."""
    V = torch.zeros(len(metas), B, T, device=dev)
    for i, meta in enumerate(metas):
        for b, m in enumerate(meta): V[i, b, :m['L']] = 1
    return V

def vq_loss(r, Ri, Gi, Gtot):
    """Quantised-state bottleneck: commitment penalty pulling each gap-step rate vector toward the nearest of K codebook vectors.
    The codebook is an exponential moving average of the rates assigned to it (no gradient; seeded from the first gap states seen).
    Biases toward a few discrete states without saying what they mean."""
    vq = r['vq']; X = Ri[Gi > 0]                                             # (n, N) rates at this chunk's gap steps
    if X.shape[0] == 0: return Ri.sum() * 0
    if vq['E'] is None: vq['E'] = X[torch.randperm(X.shape[0], device=dev)[:vq['K']]].detach().clone()
    E = vq['E']; k = torch.cdist(X.detach(), E).argmin(1)
    loss = vq['w'] * ((X - E[k]) ** 2).sum() / (Gtot * Ri.shape[-1])
    with torch.no_grad():
        for j in k.unique(): E[j] = (1 - vq['rho']) * E[j] + vq['rho'] * X[k == j].detach().mean(0)
    return loss

def step_losses(runs, U, Y, M, metas, scols, sr, si, in_mask, gen, N, alpha, soft, NINmax, micro, G=None, adapt=None, nscale=None, V=None, C=None, gam=None):
    """One training step's forward/backward in micro-batches of `micro` trials (gradients accumulate on each run's weights).
    Every term is normalised by full-batch quantities, so the summed gradient equals the full-batch gradient (noise draws aside)
    while peak memory scales with `micro`. Each chunk is run only to its longest trial (trials arrive sorted by length), which is
    exact because nothing after a trial's end enters any term. Returns the per-run full-batch loss (R,), detached."""
    B = U.shape[1]; T = U.shape[2]; nch = -(-B // micro); Ltot = torch.zeros(len(runs), device=dev); Mtot = [M[i].sum() for i in range(len(runs))]
    Gtot = [G[i].sum().clamp(min=1) for i in range(len(runs))] if G is not None else None
    if V is None: V = valid_mask(metas, B, T)
    Vtot = [V[i].sum() for i in range(len(runs))]
    Ls = np.array([[m['L'] for m in meta] for meta in metas])                 # (R,B) trial lengths
    for s in range(0, B, micro):
        sl = slice(s, s + micro); Tc = int(Ls[:, sl].max()); Ws = stack_weights(runs, NINmax)   # restacked per chunk: each backward frees its graph
        Z, R, S = forward_stack(Ws, U[:, sl, :Tc], sr, si, in_mask, gen, N, alpha, soft, adapt, nscale[:, sl, :Tc] if nscale is not None else None, gam); out = []
        for i, r in enumerate(runs):
            Zi, Yi = Z[i], Y[i][sl, :Tc]
            if r['loss'] == 'norm': base = ct.loss_norm(Zi, Yi, metas[i][sl], scols[i]) * (Zi.shape[0] / B)
            else: base = (M[i][sl, :Tc][:, :, None] * (Zi - Yi) ** 2).sum() / (Mtot[i] * c.NOUT)
            if r['hold'] and G is not None: base = base + r['hold'] * (G[i][sl, :Tc] * S[i]).sum() / (Gtot[i] * N)   # mean squared speed per unit in gaps
            if r['aux_w'] and C is not None:                                 # auxiliary running-count readout, one-hot MSE at every valid step
                Za = torch.matmul(R[i], r['W']['Wa'].T) + r['W']['cza']; Yc = torch.nn.functional.one_hot(C[i][sl, :Tc].clamp(max=15), 16).float()
                base = base + r['aux_w'] * (V[i][sl, :Tc][:, :, None] * (Za - Yc) ** 2).sum() / (Vtot[i] * 16)
            if r['vq'] is not None and G is not None: base = base + vq_loss(r, R[i], G[i][sl, :Tc], Gtot[i])
            out.append(base + r['l2r'] * (V[i][sl, :Tc][:, :, None] * R[i] ** 2).sum() / (Vtot[i] * N) + c.L2W * (Ws['J'][i] ** 2).sum() / nch)
        L = torch.stack(out); L.sum().backward(); Ltot += L.detach()
    return Ltot

def make_batches(specs):
    """Worker-process batch generation (several workers share the runs, so the data rng is seeded per run and step, not kept
    in the worker). Each spec carries the run's schedule for this step (maxsize, pre_w, pause_max, nummax, it). Returns per-run
    (U, Y, M, meta) with trials sorted by length, longest first, so micro-chunks can be truncated."""
    out = []
    for sp in specs:
        c.MAXSIZE, c.PRE_W, v.PAUSE_MAX, c.NUMMAX = sp['maxsize'], sp['pre_w'], sp['pause_max'], sp.get('nummax', 15)
        rng = np.random.default_rng([sp['seed'], sp.get('it', 0)])
        c.N, c.NIN = sp['N'], sp['NIN']; c.make = v.make_gen(**sp['gen']) if sp['gen'] else (v.make_digit if sp['var'] == 'digit' else v.make0)
        ne = int(round(sp['batch'] * sp.get('evenfrac', 0.0)))   # synth_even needs size >= blues + 1, so its blue ceiling follows the size curriculum
        items = c.synth(rng, sp['batch'] - ne) + (c.synth_even(rng, ne, vmax=min(c.NUMMAX, c.MAXSIZE - 1)) if ne else [])
        U, Y, M, meta = c.make(items, rng)
        Cn = np.zeros((U.shape[0], U.shape[1]), np.int64)                 # running count of visible blues fixated so far (auxiliary readout target)
        for i, m in enumerate(meta):
            idx = np.asarray(m['idx']); isb = np.array([s == 'b' for s in m['st']]); seen = np.zeros(len(m['st']), bool); cnt = 0
            for t in range(min(len(idx), U.shape[1])):
                j = int(idx[t])
                if j >= 0 and not seen[j]: seen[j] = True; cnt += int(isb[j])
                Cn[i, t] = cnt
            Cn[i, len(idx):] = cnt
        order = np.argsort([-m['L'] for m in meta], kind='stable')
        out.append((U[order], Y[order], M[order], [meta[j] for j in order], Cn[order]))
    return out

def J_of(r):
    """The recurrent matrix as the forward sees it: the stored J, the low-rank product Ju Jv^T, or the symmetrised J."""
    J = r['W']['Ju'] @ r['W']['Jv'].T if r.get('rank') else r['W']['J']
    if r.get('bistable'): J = J * (1 - r['eye']) + r['bistable'] * r['eye']   # fixed diagonal: no gradient reaches the learned J_ii
    return (J + J.T) / 2 if r['sym'] else J

def stack_weights(runs, NINmax):
    """Stack each run's weights on a leading axis; input weights are zero-padded to NINmax columns (padded channels carry no input)."""
    Ws = {'J': torch.stack([J_of(r) for r in runs]),
          'B': torch.stack([torch.cat([r['W']['B'], torch.zeros(r['N'], NINmax - r['NIN'], device=dev)], 1) for r in runs])}
    for k in ('c', 'W', 'cz'): Ws[k] = torch.stack([r['W'][k] for r in runs])
    return Ws

def eval_W(r):
    """Weights as the forward sees them (composed low-rank J, symmetrised J), for the single-run evaluation functions.
    A gain-annealed run (rates tanh(gam x)) is returned as the equivalent standard net with J, B, c scaled by gam
    (x' = gam x obeys the standard equations; forward_stack divides the recurrent noise by gam so x' sees the standard sigma),
    so its checkpoints are ordinary nets. Note the equivalence means a global gain is NOT a distinct bias when J is trained:
    it only rescales the effective learning rate (x gam), the weight decay and hold penalty (/ gam^2). Kept as an option, unused."""
    W = {**{k: w for k, w in r['W'].items() if k not in ('Ju', 'Jv')}, 'J': J_of(r)}; g = r.get('gam_now', 1.0)
    return {**W, 'J': W['J'] * g, 'B': W['B'] * g, 'c': W['c'] * g} if g != 1.0 else W

def weights_np(r):
    W = ct.to_np(eval_W(r))
    if r.get('rank'): W['Ju'] = r['W']['Ju'].detach().cpu().numpy().astype(np.float64); W['Jv'] = r['W']['Jv'].detach().cpu().numpy().astype(np.float64)
    return W

def schedule_for(runs, steps, it):
    """Per-run learning rate and data schedule for step it (the staged fresh recipe or the resumed decay)."""
    specs = []
    for r in runs:
        if r['fresh'] and r['staged']:
            if it <= steps // 2: pre_w = 1.0; lr = r['lr0']; maxsize = 8 if it <= steps * 3 // 16 else 14 if it <= steps * 5 // 16 else 21
            else: pre_w = 3.0; lr = 0.3 * r['lr0'] * (0.1 ** ((it - steps // 2) / max(1, steps - steps // 2))); maxsize = 21
        else: pre_w = 3.0; lr = r['lr0'] * (0.1 ** (it / steps)); maxsize = 21
        pause_max = int(r['pc_max'] * min(1.0, it / (r['pc_frac'] * steps))) if r['curric'] else 0   # pause curriculum: ceiling grows linearly
        nummax = 15
        if r['countcur']: st = r['countcur']; nummax = st[min(len(st) - 1, int(it * len(st) / max(1, steps // 2)))]   # stages span the first half
        specs.append(dict(r['spec'], maxsize=maxsize, pre_w=pre_w, pause_max=pause_max, lr=lr, nummax=nummax, it=it))
    return specs

def parse_deadline(s):
    """'' -> None; 'HH:MM' -> the next time the clock shows HH:MM (epoch seconds); otherwise epoch seconds."""
    if not s: return None
    if ':' in s:
        h, m = (int(x) for x in s.split(':')); now = time.localtime()
        t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, m, 0, 0, 0, -1))
        return t if t > time.time() else time.mktime((now.tm_year, now.tm_mon, now.tm_mday + 1, h, m, 0, 0, 0, -1))
    return float(s)

def pad_stack(arrs, T, NIN=None):
    out = []
    for a in arrs:
        p = np.zeros((a.shape[0], T) + ((NIN,) if NIN else a.shape[2:]), a.dtype); p[:, :a.shape[1], :a.shape[2]] if NIN else None
        if NIN: p[:, :a.shape[1], :a.shape[2]] = a
        else: p[:, :a.shape[1]] = a
        out.append(p)
    return torch.tensor(np.stack(out), device=dev, dtype=torch.float32)

if __name__ == '__main__':
    torch.manual_seed(0)                                                     # the global generator's only use: the VQ codebook's first pick
    if os.environ.get('MANTE_DETERMINISTIC'): torch.use_deterministic_algorithms(True)   # checks only (with CUBLAS_WORKSPACE_CONFIG=:4096:8)
    cfgs = json.load(open(sys.argv[1])); runs = [build(cfg) for cfg in cfgs]
    steps = runs[0]['steps']; batch = runs[0]['batch']; assert all(r['steps'] == steps and r['batch'] == batch for r in runs), 'steps and batch must match across the stack'
    assert len({r['N'] for r in runs}) == 1, 'all runs must have the same N'; N = runs[0]['N']; NINmax = max(r['NIN'] for r in runs)
    micro = int(next((a.split('=')[1] for a in sys.argv[2:] if a.startswith('micro=')), batch))   # micro-batch size (memory), default whole batch
    memfrac = float(next((a.split('=')[1] for a in sys.argv[2:] if a.startswith('memfrac=')), 0))  # memfrac=0.5: hard cap at that fraction of the card (OOM instead of paging)
    if memfrac: torch.cuda.set_per_process_memory_fraction(memfrac)
    rpath = f"resume_{os.path.splitext(os.path.basename(sys.argv[1]))[0]}.pt"   # training state of this stack, for resuming after a stop
    deadline = parse_deadline(next((a.split('=', 1)[1] for a in sys.argv[2:] if a.startswith('until=')), os.environ.get('MANTE_DEADLINE', '')))
    stopafter = int(next((a.split('=')[1] for a in sys.argv[2:] if a.startswith('stopafter=')), 0))
    ckpt = int(next((a.split('=')[1] for a in sys.argv[2:] if a.startswith('ckpt=')), 0))   # 1: keep learning-curve checkpoints in ckpt/
    st = torch.load(rpath, map_location='cpu', weights_only=False) if os.path.exists(rpath) and 'noresume' not in sys.argv[2:] else None
    if st is not None:
        assert st['cfgs'] == cfgs, f'{rpath} was written for a different config; delete it or pass noresume'
        if st['finished']: print(f'{sys.argv[1]}: already finished ({rpath} is marked finished)', flush=True); sys.exit(0)
    data = '../rnn/out_data.json'; test = c.real(data, 'TEST', np.random.default_rng(2))[:800]
    logs = {r['tag']: open(f"count_{r['tag']}.log", 'a' if st is not None else 'w') for r in runs}
    def say(r, s):
        print(f"[{r['tag']}] {s}", flush=True); logs[r['tag']].write(s + '\n'); logs[r['tag']].flush()
    def scores(r):
        """The 800-position test scores; skipped (not fatal) if the card is out of memory, e.g. when another process takes it."""
        try: return ct.evaluate(eval_W(r), test, r['var'], **r['kw'])
        except torch.cuda.OutOfMemoryError: pass
        torch.cuda.empty_cache(); return 'scores skipped (out of memory)'
    for r in runs:
        r['kw'] = dict(in_mask=r['in_mask'], alpha=r['alpha'], act=r['act'], adapt=r['adapt'])
        activate(r); v.PAUSE_MAX = 0
        if st is not None: continue                                       # a resumed run's header and start scores are already in its log
        say(r, f"{r['tag']} {dev} batch {batch} micro {micro} loss {r['loss']} gain {r['gain']} l2r {r['l2r']} sig {r['sig']} gen {r['gen']} hold {r['hold']} "
               f"act {r['act']} tau {'hetero' if r['alpha'] is not None else 'std'} adapt {r['adapt']} sym {r['sym']} "
               f"pausecur {(r['pc_max'], r['pc_frac']) if r['curric'] else None} gapnoise {r['gapnoise']} transfer {r['transfer']} staged {r['staged']} N {r['N']} evenfrac {r['evenfrac']} rank {r['rank']} countcur {r['countcur']} freeze {r['freeze']} lr_scale {r['lr_scale']} bistable {r['bistable']} aux {r['aux_w']} gain_anneal {r['gain_anneal']} vq {({k: v for k, v in r['vq'].items() if k != 'E'}) if r['vq'] else None} "
               f"start {scores(r)}")
    alpha = torch.stack([r['alpha'] if r['alpha'] is not None else torch.full((N,), c.ALPHA, device=dev) for r in runs])[:, None, :]
    soft = torch.tensor([r['act'] == 'softplus' for r in runs], device=dev).view(-1, 1, 1)
    adapt = None
    if any(r['adapt'] for r in runs):
        adapt = (torch.tensor([r['adapt'][0] if r['adapt'] else 0.0 for r in runs], device=dev).view(-1, 1, 1),
                 torch.tensor([r['adapt'][1] if r['adapt'] else 0.0 for r in runs], device=dev).view(-1, 1, 1))
    def extras(r):
        e = {}
        if r['alpha'] is not None: e['alpha'] = r['alpha'].cpu().numpy().astype(np.float64)
        if r['act'] != 'tanh': e['act'] = np.array(r['act'])
        if r['pulse']: e['pulse'] = np.array(True)
        if r['pos']: e['pos'] = np.array(True)
        if r['adapt']: e['adapt_beta'] = np.float64(r['adapt'][0]); e['adapt_g'] = np.float64(r['adapt'][1])
        if r['bistable']: e['bistable'] = np.float64(r['bistable'])
        if r['vq'] is not None and r['vq']['E'] is not None: e['vq_E'] = r['vq']['E'].cpu().numpy().astype(np.float64)   # the codebook, for analysis
        return e
    def keep_ckpt(r, it):
        if not ckpt or not (it <= 1000 and it % 50 == 0 or it % 250 == 0): return   # the start, every 50 steps through 1000, then every 250
        os.makedirs('ckpt', exist_ok=True)
        np.savez(f"ckpt/count_w_{r['tag']}_{it:05d}.npz", **{k: w.astype(np.float32) for k, w in weights_np(r).items()}, **extras(r))
    if st is None:
        for r in runs: keep_ckpt(r, 0)
    in_mask = torch.ones(len(runs), 1, 1, NINmax, device=dev)
    for i, r in enumerate(runs):
        in_mask[i, 0, 0, r['NIN']:] = 0
        if r['in_mask'] is not None: in_mask[i, 0, 0, :r['NIN']] = r['in_mask']
    scols = [torch.ones(c.NOUT, device=dev) for _ in runs]
    gen = torch.Generator(device=dev); gen.manual_seed(5); t0 = time.time()
    sr = torch.tensor([r['sig'][0] * np.sqrt(2 / c.ALPHA) for r in runs], device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
    si = torch.tensor([r['sig'][1] * np.sqrt(2 / c.ALPHA) for r in runs], device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
    gn = torch.tensor([r['gapnoise'] for r in runs], device=dev, dtype=torch.float32).view(-1, 1, 1)
    need_G = any(r['hold'] for r in runs) or any(r['gapnoise'] != 1 for r in runs) or any(r['vq'] for r in runs)
    need_C = any(r['aux_w'] for r in runs); need_gam = any(r['gain_anneal'] for r in runs)
    workers = int(next((a.split('=')[1] for a in sys.argv[2:] if a.startswith('workers=')), 2))
    def save_state(it, finished=False):
        """Everything a resumed run needs to continue as if uninterrupted (batches are seeded per step, so there is no data state).
        Written to a temporary file and renamed, so a kill during the write leaves the previous state intact."""
        s = dict(cfgs=cfgs, it=it, finished=finished, elapsed=time.time() - t0, gen=gen.get_state(),
                 W=[{k: w.detach().cpu() for k, w in r['W'].items()} for r in runs], opt=[r['opt'].state_dict() for r in runs],
                 vqE=[r['vq']['E'].cpu() if r['vq'] is not None and r['vq']['E'] is not None else None for r in runs])
        torch.save(s, rpath + '.tmp')
        for k in range(20):                                                  # Windows refuses the rename while another process has the file open
            try: os.replace(rpath + '.tmp', rpath); return
            except PermissionError: time.sleep(0.5)
        print(f'warning: could not replace {rpath} at step {it}; the previous state stays', flush=True)
    it0 = 0
    if st is not None:                                                       # resume: same tensors (the optimiser holds them), new values
        it0 = st['it']; t0 = time.time() - st['elapsed']; gen.set_state(st['gen'])
        for i, r in enumerate(runs):
            assert set(st['W'][i]) == set(r['W']), (r['tag'], sorted(st['W'][i]), sorted(r['W']))
            with torch.no_grad():
                for k, w in r['W'].items(): w.copy_(st['W'][i][k])
            r['opt'].load_state_dict(st['opt'][i])                          # Adam moments move to the GPU; step counts stay on the CPU
            if r['vq'] is not None and st['vqE'][i] is not None: r['vq']['E'] = st['vqE'][i].to(dev)
            say(r, f"resumed at step {it0} of {steps} from {rpath} (micro {micro}; {st['elapsed']:.0f}s trained before)")
        del st
    pool = ProcessPoolExecutor(workers) if it0 < steps else None
    if pool: specs = schedule_for(runs, steps, it0 + 1); futs = [pool.submit(make_batches, [sp]) for sp in specs]   # one step ahead, one run per task
    for it in range(it0 + 1, steps + 1):
        batches = [f.result()[0] for f in futs]
        if it < steps: nxt = schedule_for(runs, steps, it + 1); futs = [pool.submit(make_batches, [sp]) for sp in nxt]
        for r, sp in zip(runs, specs):
            for g in r['opt'].param_groups: g['lr'] = sp['lr'] * g.get('lr_scale', 1.0)
            r['gam_now'] = 1 + (r['gain_anneal'][0] - 1) * min(1.0, it / (r['gain_anneal'][1] * steps)) if r['gain_anneal'] else 1.0
        Us, Ys, Ms, metas, Cs = (list(x) for x in zip(*batches)); T = max(U.shape[1] for U in Us)
        U = pad_stack(Us, T, NINmax); Y = pad_stack(Ys, T); M = pad_stack(Ms, T); Cc = pad_stack(Cs, T).long() if need_C else None
        G = gap_mask(U, metas) if need_G else None
        nscale = (1 + (gn - 1) * G)[..., None] if G is not None and bool((gn != 1).any()) else None
        gam = torch.tensor([r['gam_now'] for r in runs], device=dev, dtype=torch.float32).view(-1, 1, 1) if need_gam else None
        while True:                                                          # out of memory (e.g. another process took the card): halve micro, redo the step
            for r in runs: r['opt'].zero_grad()
            oom = False
            try: L = step_losses(runs, U, Y, M, metas, scols, sr, si, in_mask, gen, N, alpha, soft, NINmax, micro, G, adapt, nscale, None, Cc, gam)
            except torch.cuda.OutOfMemoryError: oom = True                   # handled outside the except block, once the failed graph is released
            if not oom: break
            for r in runs: r['opt'].zero_grad()
            torch.cuda.empty_cache()
            if micro <= 16: raise RuntimeError(f'out of memory at step {it} even with micro {micro}')
            micro //= 2
            for r in runs: say(r, f"out of memory at step {it}: micro now {micro}, step redone")
        for r in runs:
            for k, fr in r['freeze'].items():
                if it <= fr * steps: r['W'][k].grad = None                 # frozen parameter: Adam skips params without a gradient
            torch.nn.utils.clip_grad_norm_([w for w in r['W'].values() if w.grad is not None], 1.0); r['opt'].step()
        if it < steps: specs = nxt
        if it % 50 == 0 or it == steps:
            for i, r in enumerate(runs):
                activate(r); c.PRE_W = 3.0
                say(r, f"{it} {round(L[i].item(), 4)} {scores(r)} {time.time() - t0:.0f}s")
                np.savez(f"count_w_{r['tag']}.npz", **weights_np(r), **extras(r)); keep_ckpt(r, it)
            save_state(it)
        if (deadline and time.time() >= deadline) or (stopafter and it - it0 >= stopafter):   # end of the time window: save and leave
            if it % 50 and it != steps:
                for r in runs: np.savez(f"count_w_{r['tag']}.npz", **weights_np(r), **extras(r))
                save_state(it)
            for r in runs: say(r, f"stopped after step {it} ({'stopafter' if stopafter and it - it0 >= stopafter else 'deadline'}, {time.time() - t0:.0f}s); a rerun resumes here")
            pool.shutdown(wait=True, cancel_futures=True)
            for f in logs.values(): f.close()
            sys.exit(3)
    if pool: pool.shutdown()
    for r in (runs if 'nofinals' not in sys.argv[2:] else []):           # nofinals: benchmarks skip the 3000-trial final scoring
        activate(r); c.MAXSIZE = 21; c.PRE_W = 3.0
        kw = r['kw']; W = eval_W(r)
        say(r, 'final (training condition) real positions    ' + ct.bybl(W, c.real(data, 'TEST', np.random.default_rng(2))[:3000], r['var'], **kw))
        say(r, 'final (training condition) balanced synthetic ' + ct.bybl(W, c.synth_even(np.random.default_rng(99), 3000), r['var'], **kw))
        c.make = v.make_gen(pulse=r['pulse'], pos=r['pos']) if (r['pulse'] or r['pos']) else v.make0 if r['var'] != 'digit' else v.make_digit
        c.SIG_REC, c.SIG_IN = 0.05, 0.02; v.PAUSE_MAX = 0
        say(r, 'final (standard schedule+noise) real positions    ' + ct.bybl(W, c.real(data, 'TEST', np.random.default_rng(2))[:3000], r['var'], **kw))
        say(r, 'final (standard schedule+noise) balanced synthetic ' + ct.bybl(W, c.synth_even(np.random.default_rng(99), 3000), r['var'], **kw))
    save_state(steps, finished=True)                                         # a rerun of this config now exits at once
    for f in logs.values(): f.close()
