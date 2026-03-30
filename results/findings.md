# Experimental Findings

## Setup

Five offline RL methods (K-Means, LSTM-DQN, RIMS, CQL-MM, CQL-SM) are evaluated on the SimBank
simulator under a step-ablation design: 1-step (RL controls only intervention 0), 2-step (interventions
0–1), and 3-step (all interventions). Each condition is run across 5 random seeds and under both
unconfounded (RCT) and confounded (CONF) logged data. Performance is reported as % gain over the
bank policy baseline.

---

## Finding 1 — Controlling more interventions consistently increases performance

Across K-Means and both CQL variants, performance improves monotonically as more interventions are
handed to the RL policy. On RCT data, K-Means goes from −119% (1-step) to +370% (2-step) to +901%
(3-step) relative to the bank baseline. CQL-MM follows a similar trajectory: −63%, +187%, +714%.
This pattern holds under both RCT and confounded data, indicating that the offline RL framework can
successfully exploit additional decision points when the learning algorithm is sufficiently stable.

---

## Finding 2 — The interest rate intervention is the most valuable decision point

Isolating the incremental gain of each additional intervention reveals that adding intervention 1
(contact HQ: 1-step → 2-step) yields roughly +250–500% incremental gain on RCT data. Adding
intervention 2 (interest rate setting: 2-step → 3-step) yields a further +380–530% for stable
methods. The interest rate decision thus drives the majority of total performance — it is the
highest-leverage point in the SimBank process. Prescriptive monitoring systems that ignore this
intervention leave most of the available value on the table.

---

## Finding 3 — Backward TD error propagation causes catastrophic failure in sequence-based methods at 3-step

LSTM-DQN and RIMS perform adequately at 2-step (+159%) but collapse at 3-step (−548%), in both
RCT and confounded conditions. The root cause is the backward TD training chain: Q3 is trained
first on terminal rewards, Q2 bootstraps from Q3, and Q1 from Q2. If Q3 learns a poor value
function — which is likely given the sparse reward signal and the complexity of encoding the full
process prefix — the estimation error cascades and amplifies through Q2 into Q1. K-Means and CQL
methods are not affected: K-Means uses fitted Q-iteration with fixed tabular targets, and CQL's
conservative penalty regularizes Q-value overestimation throughout.

---

## Finding 4 — Confounding reduces performance but preserves the relative ordering of methods

On RCT data, policies achieve higher absolute gains. On confounded data, the bias in the logged
actions leads to lower quality learned policies — K-Means drops from +901% to +600%, CQL-SM from
+723% to +569%. Crucially, the *ranking* of methods and the *directional pattern* across steps are
preserved between conditions. This provides evidence that the step-ablation findings generalize
beyond the RCT setting.

---

## Finding 5 — Seed variance increases with task complexity and is method-dependent

At 1-step and 2-step, all methods show low inter-seed variance — results are stable across the five
random seeds. At 3-step, the picture diverges sharply. K-Means achieves the highest average gain
but also the widest interquartile range (roughly 600–1000% on RCT), suggesting sensitivity to data
splits and initialization. CQL-MM and CQL-SM show narrower distributions, indicating more stable
learning despite the more complex task. LSTM-DQN and RIMS show near-zero variance — but only
because all seeds converge to the same degenerate policy, not because the method is reliable.

---

## Finding 6 — Controlling only the first intervention is insufficient for all methods

No method reliably beats the bank policy in the 1-step setting. This is not a failure of the
methods per se — it reflects that the procedure choice (standard vs. priority path) alone has
limited impact on the final outcome when downstream decisions are left to the bank. This finding
has a practical implication: deploying a prescriptive system that targets only the first decision
point provides little to no benefit over the existing bank policy.

---

## Scope of Claims

| Can claim | Cannot claim |
|---|---|
| More interventions = more value (robust across 5 seeds, 2 conditions) | K-Means is fundamentally better than LSTM/RIMS |
| LSTM/RIMS instability at 3-step due to backward TD error propagation | CQL is better than K-Means in general |
| Interest rate setting is the highest-value intervention in SimBank | These rankings generalize to other datasets or process structures |
| Confounding reduces gains but preserves method rankings | |
| CQL methods show more stable learning than K-Means at 3-step | |
