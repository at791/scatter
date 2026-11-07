# Scatter PFC RNN

A rate recurrent network, in the style of Mante, Sussillo, Shenoy & Newsome (2013), that plays the counting game **Scatter**, with live views of its dynamics under the board.

Play it: open `rnn/scatter_pfc.html` (or `index.html`, which forwards to it). With GitHub Pages switched on for this repository, the game is served at the repository's Pages URL. The page is a single self-contained HTML file: the weights and all panel data are embedded, and nothing is loaded from the network.

## The game

Scatter is a board of stones. Some stones carry a number; that number says how many of the stones in a circle around it are blue. Stones start hidden, and the player's job is to work out the colour of each hidden stone, blue or dark, from the numbers. For one circle there are three possible verdicts:

- **rest dark**: the visible blues already reach the number, so every hidden stone in the circle is dark;
- **rest blue**: number minus visible blues equals the number of hidden stones, so they are all blue;
- **can't tell**: anything in between.

## The network

- 300 tanh rate units, `x <- (1 - a) x + a (J tanh x + B u + c)`, dt = 20 ms. Half the units have a time constant of 100 ms, the other half time constants drawn log-uniformly between 1 and 5 s (`a = dt / tau` per unit).
- Inputs (5): a 60 ms onset pulse on the blue, dark or hidden channel when the network looks at a stone, the circle's number (tonic, number / 10), and a "scan complete" signal (tonic after the last stone).
- Outputs (3): rest dark, rest blue, can't tell. They stay near 0 until the network has an answer; the page acts as soon as one crosses 0.5 (free response).
- Training: numpy or PyTorch back-propagation through time with Adam, on synthetic circles scanned one stone at a time (fixations 200-300 ms, gaps 0-300 ms, occasional long interruptions), with unit and input noise, a light L2 penalty, a penalty on drift during gaps, a circle-size curriculum and a share of each batch balanced across high blue counts. The shipped checkpoint is `w_base` of `mante/stack3a.json` (6000 steps, batch 512).
- Accuracy of the shipped checkpoint: 98.9% on held-out real game positions and 97.6% on a balanced synthetic set at training noise; 100% with noise off. The page runs it without noise.

How it works, in short: the number input pushes the state out along a counting axis, each blue stone steps it back, and the verdict fires when the remaining need reaches zero. The count is held on a graded line, not in discrete attractors. `docs/counting_synthesis.md` summarises the eight training stacks that established this, and `docs/counting_literature.md` places it against the counting-network literature.

## What the page shows

Press "Watch the RNN play". The network takes one circle at a time, looks at its stones in turn from the top, and marks the hidden stones when an output crosses 0.5. A playback bar lets you pause, slow down, and step stone by stone, forwards and backwards within a circle. Under the board, six panels drawn from the network's own activity:

1. **The troughs in 3-D**: the slow manifolds of all 15 numbers in the first three principal components of the count states, with the state moving on the current number's trough (drag to rotate).
2. **Trough and verdicts**: the current number's trough unrolled, against the direction the state takes toward its verdict, with the three verdict regions.
3. **Speed landscape across the trough**: the cross-section at the current count position, spanned by the trace a hidden stone leaves and the push of a dark stone, with the network's speed and flow there.
4. **What the network sees**: the five input channels through the current circle.
5. **The three output units** through the current circle.
6. **Real population plane**: mean activity of the need-type (81) and slack-type (23) units, groups found by fitting each unit's end-of-scan activity on real positions; ticks mark where real positions with each tally sit.

Nothing in the panels is a fitted decoder of task variables: axes are principal components, input pushes, measured traces, or unit groups, and the traces are the raw state.

## Repository layout

```
index.html                 forwards to the game
rnn/
  scatter_pfc.html         the game with the network and the panels (built; 1.9 MB)
  scatter_base.html        the game without a network: the input to the build
  count_ai.js              the JS port of the network, the player and the six panels
  splice_count.py          splices count_ai.js and the panel data into the page
  rebuild.py               base page -> labels -> unit groups -> splice (run inside rnn/)
  out_data.json            real game positions (train / test split) used for evaluation and unit groups
mante/
  count.py                 the task, the model, numpy training and evaluation (100-unit CPU version)
  count_var.py             model and generator variants (pulses, pauses, positions, wider nets, ...)
  count_torch.py           PyTorch port of the training loop
  count_multi.py           trains a stack of configurations in one GPU process; stack*.json are the stacks
  count_pre.py, count_succ.py   pretrained counters (cardinal and successor-function targets)
  count_lstm.py            LSTM control
  count_finals.py, count_curves.py, count_learnspeed.py   scoring and learning curves
  count_w_w_base.npz       the shipped checkpoint (J, B, c, W, cz, per-unit alpha)
  count_pack.py            checkpoint -> count_b64.txt for the page
  count_frames2.py         unit groups and ticks for the population plane -> count_frames_w.json
  count_landscape_export.py, count_rib3d_export.py, count_trough_export.py, count_cross_export.py
                           panel data -> count_land.json, count_rib3d.json, count_trough.json, count_cross.json
  viz_export.ps1           runs the six export steps for another checkpoint
  count_attr.py, count_attr_eig.py, count_halfkick.py, count_landscape.py, count_cueswap.py, count_wells_viz.py
                           dynamics measurements: pause tolerance, slow points, half-kick (line or wells), cue swaps
  count_w_skewed.npz       the 100-unit game net that the continuation stacks start from
  count_w_pre.npz, count_w_pre300.npz, count_w_succ300.npz   pretrained counters (cardinal 100 / 300 units, successor 300)
  count_w_lstm.pt          the LSTM control
docs/
  counting_synthesis.md    what the training stacks found
  counting_literature.md   the counting-network literature this was compared against
```

## Rebuilding the page

The repository already holds the built page and every intermediate file, so this is only needed after changing `count_ai.js`, the base page, or the checkpoint.

```
cd rnn
python rebuild.py
```

To swap in another checkpoint, first regenerate the embedded data in `mante/` (CPU, a few minutes; the exports read the checkpoint as `count_w_w_base.npz` from the working directory):

```
cd mante
python count_pack.py count_w_w_base.npz count_b64.txt
python count_frames2.py ../rnn/out_data.json count_w_w_base.npz count_frames_w.json
python count_landscape_export.py count_w_w_base.npz count_land.json
python count_rib3d_export.py
python count_trough_export.py
python count_cross_export.py
cd ../rnn
python rebuild.py
```

`count_ai.js` is written for a 300-unit network with per-unit time constants (`CN_N`); a checkpoint of another size needs that constant and the shapes changed.

## Training

The 100-unit CPU version (numpy only; `STEPS` Adam steps, batch 64):

```
cd mante
python count.py 6000 ../rnn/out_data.json
```

The shipped 300-unit network and its siblings (PyTorch, one GPU, roughly 1 s per step for a stack of four):

```
cd mante
python count_multi.py stack3a.json micro=128
```

`micro` is the number of trials per forward/backward chunk; lower it if memory runs out. Checkpoints are written as `count_w_TAG.npz` with the numpy keys, so every analysis script reads them. `stack1a.json` to `stack8b.json` are the other stacks described in `docs/counting_synthesis.md`.

## Running locally

Open `rnn/scatter_pfc.html` directly in a browser, or serve the folder:

```
python -m http.server 8000
```

and open http://localhost:8000/.

## Requirements

Python 3.10+ with numpy; PyTorch for the GPU training scripts and the LSTM control; matplotlib for the figure scripts. See `requirements.txt`.

## References

- Mante, V., Sussillo, D., Shenoy, K. V. & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. *Nature* 503, 78-84.
- Sussillo, D. & Barak, O. (2013). Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. *Neural Computation* 25, 626-649.

## License

MIT, see `LICENSE`.
