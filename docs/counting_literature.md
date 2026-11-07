# Counting and the cardinal principle in neural networks: what transfers to the serial-scan RNN

Compiled 2026-10-30 from five parallel literature sweeps (cardinal-principle models; numerosity codes; counting in RNNs/transformers; neuroscience of serial counting; sequential-attention object counting). Full texts were read unless marked otherwise in the sweep reports. Extracted texts are in the session scratch folders `texts/`, `txt/`, `pdftext/`.

## 1. Convergent diagnosis of our network

Our 100-unit rate RNN implements a Meck and Church accumulator in "run mode": a graded leaky line attractor that integrates the stone input over dwell time, with noise that grows with the count. Every strand independently reports the same ceiling for graded codes:

- Kondapaneni and Perona 2020: a line-like embedding trained with no number labels forms separable islands only for 0 to 6 or 8 objects, then merges.
- Zhang and Wu 2020 (NeurIPS): a count-supervised CNN under distribution shift is "high up to 4, then it drops"; the estimation interval is 1 for N ≤ 4 and 3 to 4 above 14.
- Dasgupta, Hattori and Navlakha 2022: multiplicative/compressive counting codes separate "1-2-3-many" (fly MBON data).
- Dulberg, Webb and Cohen 2021: plain LSTM and transformer give-N agents "had good performance until around N = 5, past which performance degraded".
- Verma and Sengupta 2023: a recurrent inhibition level that codes 1 to 4 monotonically cannot cover 5 to 17.
- Miller 2013: a depression-driven attractor counter distinguishes 9 counts at low noise but only 3 at higher noise.
- Cordes et al. 2001: accumulator (nonverbal) counting has scalar variability at every N; verbal counting has binomial variability because each item is a discrete step.

Chen and Verguts 2013: summation (monotonic) coding "is more suited for smaller-larger processing", numerosity-selective coding "is more efficient for same-different number discrimination"; without number-relevant training "only summation coding would be observed". Our objective is a magnitude comparison scored once at the end, so a summation code is the predicted outcome.

Formal results (Weiss et al. 2018; Merrill et al. 2020; Korsky and Berwick 2019; Hewitt et al. 2020): a bounded tanh RNN is finite-state, so it cannot count unboundedly, but a count bounded at 15 is regular and needs only O(15) saturated units, i.e. 15 discrete attractors. Gradient descent instead finds the approximate line attractor (Maheswaranathan et al. 2019), which extrapolates about 2x the training range (Wiles and Elman 1995; El-Naggar et al. 2022) and whose exactness conditions "do not coincide with the minimal loss value" and are "unlearned during training" (El-Naggar 2023). Delétang et al. 2023: tanh RNNs generalise perfectly to 12x length on cyclic/modular tasks but fail monotone accumulation.

## 2. What every successful exact counter has

1. One discrete event per item (presence/commit gate in AIR, SPACE, IRLC; "event mode" pulse in Meck and Church; entity gate in Sabathiel et al. 2020).
2. A number-independent successor step (Dulberg's successor chain; Sabathiel 2022 abacus; Curto's CTLN ring counter; Koulakov's bistable staircase).
3. A memory of visited items that is indexed by location, not by magnitude (Sabathiel's emergent ConvLSTM "mentally tagged" cell; Thompson et al. 2023/2024 map layer; Dulberg's environment removing selected items; Ren and Zemel's canvas).
4. A relational stop/match decision (dot-product between running state and cue; Dulberg) rather than a threshold on a magnitude.
5. Counts separated by fixed margins that do not shrink with N (Behrens et al. 2025; Barbero et al. 2024 representational collapse; Zhang, Hare and Prügel-Bennett 2018: never normalise the accumulator).

None of these mechanisms requires count labels; several papers used labels for training, but the mechanism itself is architectural or self-supervised.

## 3. Integrable approaches, deduplicated and ranked

Grouped by where the change goes. "Done" = already implemented in our stacked trainer.

### Input
- I1. Onset-pulse stone input (Meck and Church event mode; Nieder duration-invariant neurons). Done. Removes the dwell-time confound; alone still Weber-limited.
- I2. Position input: the index or angle of the fixated stone, so repeated identical items are individuated (Thompson 2023/2024; Sabathiel 2020; Cheyette et al. 2024 FINST account). Trivial.
- I3. Successor-encoded number cue: cue N = the N-th iterate of a fixed rotation in a small subspace, instead of N/10 or one-hot (Dulberg 2021). Gives the order of numerals for free (our one-hot "digit" variant had to learn it and lagged) and lets the same per-blue operator reach the cue. Trivial as an input change.

### Objective (label-free)
- O1. Hold penalty on deterministic state speed in gaps (Haviv et al. 2019; TCR 2025). Done.
- O2. Self-supervised map auxiliary: a 21-slot head predicting "a blue was seen at slot k so far", target derived from the input stream (Thompson: "the map loss is necessary for accurate counting"). Converts the count into a sum over slots. Low cost; needs I2.
- O3. Margin on adjacent counts: hinge so n and n+1 decisions are separated by a fixed gap; over-sample minimal pairs at large n (Behrens 2025; Dulberg). Low cost.
- O4. Confidence penalty pushing per-item contributions to 0 or 1 and a snap-to-integer readout (Zhang, Hare and Prügel-Bennett 2018). Low cost; needs a per-item gate (A3).

### Architecture and dynamics
- A1. Heterogeneous or learned time constants. Done (fixed); learned τ next (Kim and Sejnowski 2021).
- A2. Per-unit adaptation or synaptic depression with strong self-excitation: an itinerant attractor counter where each identical item hops the state to a new fixed point (Miller 2013; Chen and Miller 2020). One slow variable per unit. Must be tuned for 0 to 300 ms gaps (untested in the papers).
- A3. Per-fixation Bernoulli commit gate, Gumbel-sigmoid relaxed with temperature annealing and a sparsity prior; count = sum of gates (AIR; Discrete-AIR; SPACE; Figurnov PACT). Medium cost.
- A4. Bistable or hysteretic units with a ratchet threshold: count = number of switched units, drift-free, tolerant to ±20% mistuning (Koulakov et al. 2002; Nikitchenko and Koulakov 2008; Hewitt 2020). Must be paired with I1 or it integrates dwell again.
- A5. One or two unsquashed accumulator units with recurrent gain pinned to 1 plus a late-onset integer regulariser on the increment weights (Weiss 2018; NAC/NAU, Madsen and Johansen 2020; El-Naggar: gain exactly 1 and matched increments are necessary and sufficient). Low architectural cost; the regulariser is what makes it exact.
- A6. Ring or chain counter prior: modular code (quotient/remainder rings; Merrill 2020; Curto CTLN counter; Fan et al. 2024 input-dependent rotations). Tanh RNNs are reliable on cyclic transitions. Needs initialisation or constraint.
- A7. Negative-derivative feedback (Lim and Goldman 2013) to kill the seconds-scale leak without discreteness. Secondary.
- A8. Visited-item latch / inhibition of return (Ullman marking; IRLC's additive negative logit; SQAIR's conditioning on already-explained set). Only needed if the scan can revisit; valuable with I2.

### Training protocol
- T1. Pause curriculum: delays introduced gradually and extended to tens of seconds, with per-period loss normalisation (Haviv's DeCu; Ságodi 2024). Code exists for pauses; curriculum not yet.
- T2. Count curriculum with persistent rehearsal of small N, flat count distribution, train beyond the tested range (Fang 2018 trained to 15, tested to 9; Dulberg; Testolin 2020). Partly in place.

## 4. Three coherent routes

- Route S (slot/map): I1 + I2 + O2 (+ I3, O1). Count becomes a sum over a location-indexed map; the integrator is no longer asked to hold a magnitude.
- Route A (attractor chain): I1 + A2 or A4 (+ O1, A1). Each item hops the state to the next discrete attractor; the hold penalty and slow units stabilise the states.
- Route E (exact accumulator): I1 + A5 + O3 (+ I3). A linear counter with pinned gain and integer increments, read out by margin against a successor-coded cue.

Each route is testable as 2 to 3 runs in one stack; all keep the latent count unsupervised.

## 5. Key references
Fang, Zhou, Chen, McClelland 2018 CogSci; Sabathiel, McClelland, Solstad 2020 CogSci (two papers); Sabathiel et al. 2022 NLDL; Dulberg, Webb, Cohen 2021 CogSci (arXiv 2105.10577); Thompson, Sheahan, Summerfield 2023 PMLR 210; Thompson et al. 2024 Neuron; Kondapaneni, Perona 2020 (arXiv 2012.04132); Rane et al. 2024 CogSci; Gupta, Varma 2024 (arXiv 2311.15194); Pecyna et al. 2019/2022; Hein, Diepold 2024 Cogn Sci; Dehaene, Changeux 1993; Verguts, Fias 2004; Stoianov, Zorzi 2012; Testolin, Zou, McClelland 2020; Chen, Verguts 2013; Nasr, Viswanathan, Nieder 2019; Kim et al. 2021; Zhang, Wu 2020 NeurIPS; Mistry et al. 2023 Nat Commun; Nieder, Merten 2007; Roitman, Brannon, Platt 2007; Nieder 2012 PNAS; Ditz, Nieder 2020; Cordes et al. 2001; Meck, Church 1983; Grant, Goodman, McClelland 2024/2025; Strock, Tan, Menon 2026 CogSci; Verma, Sengupta 2023; Weiss, Goldberg, Yahav 2018; Suzgun et al. 2019; Merrill 2020; Merrill et al. 2020; Hewitt et al. 2020; Delétang et al. 2023 ICLR; Chang, Bisk 2025 ICLR; Yehudai et al. 2024; Ouellette et al. 2023; Barbero et al. 2024; Behrens et al. 2025 ICML; Trask et al. 2018 NALU; Madsen, Johansen 2020 ICLR; El-Naggar et al. 2022/2023; Maheswaranathan et al. 2019; Wiles, Elman 1995; Rodriguez, Wiles 1997; Korsky, Berwick 2019; Miller 2013; Chen, Miller 2020; Koulakov et al. 2002; Nikitchenko, Koulakov 2008; Lim, Goldman 2013; Cain et al. 2013; Dasgupta, Hattori, Navlakha 2022; Londono Alvarez, Morrison, Curto 2025 (CTLN counter); Melcher, Piazza 2011; Trick, Pylyshyn 1994; Peterson, Simon 1997; Eslami et al. 2016 AIR; Kosiorek et al. 2018 SQAIR; Wang, Jamnik, Liò 2019 Discrete-AIR; Lin et al. 2020 SPACE; Crawford, Pineau 2019 SPAIR; Trott, Xiong, Socher 2018 IRLC; Zhang, Hare, Prügel-Bennett 2018; Chattopadhyay et al. 2017; Ren, Zemel 2017; Ranjan et al. 2021; Liu et al. 2022 CounTR; Haviv, Rivkind, Barak 2019; Ságodi et al. 2024; Driscoll, Shenoy, Sussillo 2024; Kilpatrick, Ermentrout, Doiron 2013; Kim, Sejnowski 2021.
