# Counting in the Scatter RNNs: a line, not wells

Synthesis of training stacks 1 to 8, 30 October to 3 November 2026. All networks play the serial-scan counting game: stones arrive one at a time as onset pulses, the circle's number is a steady input, and the network must answer rest blue, rest dark or can't tell. Accuracies are on the balanced test set at standard noise unless stated.

## Short answer

- Every network trained on the game keeps its count on a graded line, not in point attractors. This holds for unbiased nets, noise- and pause-trained nets, transfers from pretrained counters, and nets biased toward point attractors.
- Lines are enough. The best rate network answers every one of 9000 test trials correctly once noise is switched off.
- At standard noise the best rate networks score 98.4 to 98.5%. The remaining errors come from noise acting on the decision, not from how the count is stored. Reaching 100% at standard noise is therefore a readout question.
- Wells appear only in networks pretrained on counting itself, and game training removes them. A successor-function prior gives a small head start in learning; a cardinal counting prior gives none.

## How the tests read wells versus a line

- **Half-kick test.** The state is placed part of the way between two adjacent count states and run in silence for 10 s. Wells pull it to the nearer count; a line keeps it where it was put, or lets it drift proportionally. The test was validated on the pretrained counters, which snap.
- **Eigenvalues at slow points.** With heterogeneous time constants these cannot separate a well from a slow leak, so they were never decisive.
- **Functional tests.** Holding through inserted pauses, the decay of a single blue's push, and the spacing of count states.

## What each stack showed

| Stack | What changed | Result |
|---|---|---|
| 1 | 100 units: onset pulses, slow units, hold penalty, adaptation, position, pause curriculum, symmetric weights | Pulses with a third of units slow lifted counts above 4 from 74 to 86–87%. The rest did not help. |
| 2 | 100 units: half the units slow, softplus, gap noise, self-excitation, first pretrained-counter transfer | Half slow units gave the best 100-unit net, 92.6%. The pretrained counter's wells were lost in game training. |
| 3 | 300 units, extra high-count circles, long pauses, training noise, count curriculum, low rank; LSTM control | 97.6–98.4% across seeds; 100% with noise off. Long pauses let nets hold through 10 s of silence at 98–99%. LSTM 98.9%, also a line. |
| 4 | Evidence: three seeds of the best conditions, training noise 1.5 to 4 times | All conditions equal within about a point. Noise up to 2 times costs nothing; 3 and 4 times cost 4 and 10 points. All lines. |
| 5 | Cardinal pretrained counter transferred: free, slowed, part-frozen, frozen, with pauses | Free transfer 98.2%, equal to fresh nets. Every restriction costs accuracy; frozen with long pauses fails at 55%. All lines. |
| 6 | Point-attractor biases only: fixed self-coupling, self-excitation, a pull toward 64 learned points | Bistable units make the state restless, never settling, at 81–92%. The pull toward learned points gives the stillest lines, at 97–98%. No wells. |
| 7 | Successor-function pretraining: each blue advances a code by one fixed rotation | A better counter than the cardinal one, 99.9% at gap ends and perfect to 12 blues. Transfer 98.5%. Wells dropped again. |
| 8 | Learning speed: fresh vs cardinal vs successor transfers, with and without training aids | Successor transfers reach 90% sooner in most comparisons; cardinal transfers no sooner. All end in the same place. |

## Why the line wins

The game's own solution makes the number a live offset: the steady number input pushes the state along one continuous count axis, each blue steps it back, and the readout fires when the state reaches the point where the remaining need is zero. States from different numbers that share the same remaining need sit next to each other. A count held in wells resists exactly this shift.

- A number input of the size the game uses tilts a pretrained counter's wells away.
- With all weights free, game training writes a line into the recurrent weights with a change of about 31% that sits in a handful of directions. A tenth of that change already removes the snapping.
- With the recurrent weights frozen, training moves the operating point through the biases and input weights instead, and the frozen counter's own weights then hold a line too.

Nothing in the game rewards keeping the blue count apart from the number, so a small change turns wells into the line the readout needs. The only switch-like behaviour in any game net is the final decision near zero remaining need.

## Pretrained networks

- **Kept?** No. Both the cardinal counter and the successor net have real wells before transfer, and every game-trained version of them is a line.
- **Better at the end?** No. Free transfers match fresh nets: 97.7–98.2% for the cardinal and 98.3–98.5% for the successor transfers, against 97.6–98.4% for fresh nets. Protecting the prior only costs accuracy.
- **Faster?** The successor prior gives a modest, consistent head start: on the balanced set it reaches 90% at 1500–2000 steps against 1750–3250 for fresh nets, and without training aids at 1750–2000 against 2500–3250. The cardinal prior gives no reliable speedup.
- **A substitute for training aids?** No. Without the curriculum and the extra high-count circles every network, pretrained or not, ends at 94–96% instead of 98%, mostly losing circles with 11 or more blues.

## Biasing toward point attractors

Unit-level bistability does not add up to count-level wells: those networks never settle in silence, their count codes are compressed or jumbled, and they lose accuracy and memory through pauses. A pull toward 64 learned points is satisfied by spacing the points along the line, giving the stillest lines of all. A pull toward exactly 16 points is untested.

## Accuracy: how close to 100%

| Network | Balanced set, standard noise | Noise off |
|---|---|---|
| Best rate networks: successor transfers, fresh 300-unit seeds | 98.3–98.5% | 100% for w_base, 0 errors in 9000 trials |
| LSTM control, 100 units | 98.9% | 99.9% |
| First serial network, 100 units, standard time constants | about 68% | |

At standard noise the count memory is intact; errors come from noise moving the decision readout, as wrong answers and some early commits. In the LSTM, most wrong answers were on can't-tell circles, and a simple persistence rule removed the early commits but capped out at 99.4%. Closing the last 1.5% means changing how the decision is read from a noisy state, for example accumulating evidence over time before committing, or training for a wider decision margin. That is untested.

## What helped and what did not

- **Helped:** onset pulses for stones; slow units with time constants of 1 to 5 s; width 300 with extra high-count circles; training with long pauses for holding through silence.
- **Made no difference:** the hold penalty, a count curriculum, training noise up to 2 times, the pull toward learned points, and pretraining for final accuracy.
- **Hurt:** adaptation, symmetric weights, low-rank weights, training noise of 3 times and more, self-excitation and fixed bistable units, and freezing pretrained weights.

## Open threads

- **100% at standard noise** is a readout question: how the decision is read from a noisy state.
- **When the pretrained wells disappear** during transfer can now be tested on the learning-curve checkpoints saved every 50 steps in stack 8.
- **A pull toward exactly 16 points** is the one point-attractor bias not yet tried.
- **Interactive visualisations** of what each model change does remain on the to-do list.

Data, logs and scripts are in the mante folder the app made for this session; this file is also saved there as synthesis.md.
