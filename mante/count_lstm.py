"""LSTM control for the counting game.  python count_lstm.py [steps=3500] [batch=512] [H=100] [evenfrac=0.5] [tag=lstm]
Same data as the rate nets (pulse generator, size curriculum, a share of each batch balanced on visible blues), same masked-MSE loss
on the three outputs, same decision rule and by-blue scoring. Noise is matched to count.forward: input noise on all channels and
additive Gaussian noise on the hidden and cell states at each step, both at the rate net's per-step standard deviation.
Weiss, Goldberg & Yahav 2018: LSTMs learn exact counters (a level in the cell state behind a saturated forget gate) where simple
RNNs find analog ones. Writes count_TAG.log and count_w_TAG.pt."""
import os; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, time, numpy as np, torch, count as c, count_var as v, count_torch as ct
dev = ct.dev
kw = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
steps = int(kw.get('steps', 3500)); batch = int(kw.get('batch', 512)); H = int(kw.get('H', 100)); tag = kw.get('tag', 'lstm'); evenfrac = float(kw.get('evenfrac', 0.5))
lr0 = float(kw.get('lr', 1e-3)); train_noise = kw.get('noise', '1') != '0'
c.make = v.make_gen(pulse=True); v.PAUSE_MAX = 0; c.NIN = 5
torch.manual_seed(1); rng = np.random.default_rng(1)
cell = torch.nn.LSTMCell(c.NIN, H).to(dev); W = torch.nn.Linear(H, c.NOUT).to(dev)
TMAX = int(kw.get('tmax', 600))                                   # chrono init (Tallec & Ollivier 2018): forget-gate biases log U(1, Tmax) so memory spans 1..Tmax steps
with torch.no_grad():                                             # (a bias of 1 gives f = 0.73, a 60 ms memory at 20 ms steps; the trials are ~600 steps)
    bf = torch.log(torch.empty(H).uniform_(1.0, float(TMAX))); cell.bias_ih[H:2 * H] = bf.to(dev); cell.bias_hh[H:2 * H].zero_()
    cell.bias_ih[:H] = -bf.to(dev); cell.bias_hh[:H].zero_()      # input gate = 1 - forget gate at init
params = list(cell.parameters()) + list(W.parameters()); opt = torch.optim.Adam(params, lr=1e-3)

def forward(U, noise=True, gen=None):
    """U: (B,T,NIN). Returns Z (B,T,NOUT) and the hidden states (B,T,H). Noise goes on the hidden state only: the cell state
    has no leak, so additive noise there would random-walk without bound, a harsher regime than the rate net's leaky x."""
    B, T, _ = U.shape; sr = c.SIG_REC * np.sqrt(2 / c.ALPHA) * c.ALPHA if noise else 0; si = c.SIG_IN * np.sqrt(2 / c.ALPHA) if noise else 0
    if si: U = U + torch.randn(U.shape, device=dev, generator=gen) * si
    h = torch.zeros(B, H, device=dev); cs = torch.zeros(B, H, device=dev); hs = []
    Us = U.unbind(1)
    for t in range(T):
        h, cs = cell(Us[t], (h, cs))
        if sr: h = h + torch.randn(h.shape, device=dev, generator=gen) * sr
        hs.append(h)
    Hs = torch.stack(hs, 1)
    return W(Hs), Hs

def run(items, seed):
    U, Y, M, meta = c.make(items, np.random.default_rng(seed), interrupt=False); gen = torch.Generator(device=dev); gen.manual_seed(seed + 1)
    with torch.no_grad(): Z, _ = forward(torch.tensor(U, device=dev), True, gen)
    return Z.cpu().numpy(), meta

def evaluate(items):
    Z, meta = run(items, 0); ok = early = never = 0; rts = []
    for i, m in enumerate(meta):
        d = v.decide(Z[i, :m['L']], 'base')
        if d is None: never += 1; continue
        if d[0] < m['det']: early += 1; continue
        if d[1] == m['lab']: ok += 1; rts.append((d[0] - m['det']) * 20)
    n = len(meta); return dict(acc=round(ok / n, 4), early=round(early / n, 4), never=round(never / n, 4), rt_ms=round(float(np.median(rts)), 0) if rts else None)

def bybl(items, seed=0):
    Z, meta = run(items, seed); ok = np.zeros(len(meta), bool)
    for i, m in enumerate(meta):
        d = v.decide(Z[i, :m['L']], 'base')
        if d and d[0] >= m['det']: ok[i] = d[1] == m['lab']
    nb = np.array([m['st'].count('b') for m in meta]); lab = np.array([m['lab'] for m in meta])
    return (f'all {ok.mean():.3f} <=4 {ok[nb <= 4].mean():.3f} >4 {ok[nb > 4].mean():.3f} by label {[round(float(ok[lab == l].mean()), 2) for l in range(3)]} '
            f'by blues 0..10,11+ {[round(float(ok[nb == b].mean()), 2) for b in range(11)] + [round(float(ok[nb >= 11].mean()), 2)]}')

if __name__ == '__main__':
    data = '../rnn/out_data.json'; test = c.real(data, 'TEST', np.random.default_rng(2))[:800]
    log = open(f'count_{tag}.log', 'w')
    def say(s): print(s, flush=True); log.write(s + '\n'); log.flush()
    say(f'{tag} {dev} lstm H {H} batch {batch} steps {steps} lr {lr0} train_noise {train_noise} evenfrac {evenfrac} sig {c.SIG_REC},{c.SIG_IN} gen pulse start {evaluate(test)}')
    gen = torch.Generator(device=dev); gen.manual_seed(5); t0 = time.time()
    for it in range(1, steps + 1):
        if it <= steps // 2: c.PRE_W = 1.0; lr = lr0; c.MAXSIZE = 8 if it <= steps * 3 // 16 else 14 if it <= steps * 5 // 16 else 21   # the staged fresh recipe
        else: c.PRE_W = 3.0; lr = 0.3 * lr0 * (0.1 ** ((it - steps // 2) / max(1, steps - steps // 2))); c.MAXSIZE = 21
        for g in opt.param_groups: g['lr'] = lr
        ne = int(round(batch * evenfrac)); items = c.synth(rng, batch - ne) + (c.synth_even(rng, ne, vmax=min(15, c.MAXSIZE - 1)) if ne else [])
        U, Y, M, meta = c.make(items, rng); U, Y, M = (torch.tensor(a, device=dev) for a in (U, Y, M))
        Z, Hs = forward(U, train_noise, gen)
        L = (M[:, :, None] * (Z - Y) ** 2).sum() / (M.sum() * c.NOUT) + c.L2R * (Hs ** 2).mean()
        opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if it % 50 == 0 or it == steps:
            c.PRE_W = 3.0
            say(f'{it} {round(L.item(), 4)} {evaluate(test)} {time.time() - t0:.0f}s')
            torch.save(dict(cell=cell.state_dict(), W=W.state_dict(), H=H), f'count_w_{tag}.pt')
    c.MAXSIZE = 21; c.PRE_W = 3.0
    say('final (standard schedule+noise) real positions    ' + bybl(c.real(data, 'TEST', np.random.default_rng(2))[:3000]))
    say('final (standard schedule+noise) balanced synthetic ' + bybl(c.synth_even(np.random.default_rng(99), 3000)))
    log.close()
