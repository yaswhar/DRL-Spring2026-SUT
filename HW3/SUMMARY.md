# HW3 Summary

A short note on **what** was built, **how**, **why**, **what we reached**, and **what it means**.

## What
**Deterministic actor-critic** on a moving-target MuJoCo `Reacher`: build **DPG → DDPG → TD3** and ablate TD3's three additions.

## How
Deterministic `tanh`-squashed actor + Q-critic; one-step **DPG**; then **DDPG** adds a **replay buffer + target networks + OU exploration**; then **TD3** adds **clipped double-Q**, **target-policy smoothing**, and **delayed actor/target updates**. Bootstrapping masks true termination only. An evaluation cell reports each agent's deterministic return and mean **fingertip-target distance**.

## Why (two real fixes)
Careful tracing, rather than simply checking that the code runs, surfaced two pitfalls in the continuous actor-critic:
1. **Actor `tanh` saturation, leading to a dead policy gradient.** Large pre-activations rail `tanh` to ±1, the gradient vanishes, and training goes **flat**. Fixed with a **small final-layer init** (`±3e-3`) so initial actions start near 0.
2. **Uncontrolled per-agent randomness.** `train_agent` reseeded only the env, so comparisons measured init luck. Fixed with **`set_global_seed()` before each agent**.

## What we reached (real training)
25k steps/agent, seeded (see `figures/ablation_curves.png`); per-agent eval (return and fingertip distance) is produced by the **Evaluation Summary** cell:

| Agent | Eval return | Fingertip error |
|---|---|---|
| DPG | −25.8 | ~15 cm (never tracks) |
| DDPG | −1.99 | ~1.3 cm |
| **TD3 (full)** | **−1.72** | **~1.1 cm (best)** |

Ablations (train return, last 20): full **−2.82**; no-smoothing **−3.45**; no-delay −2.89; no-double-Q −2.86.

## What it means
- **Replay buffer + target networks are decisive** (the DPG → DDPG jump of about +24 return); all TD3 variants track to roughly 0.6 to 1.1 cm.
- On this *easy, densely rewarded* task TD3's conservatism is barely needed: by deterministic eval, **removing clipped double-Q tracked best** (5.7 mm versus 11 mm for full TD3), because with little overestimation to correct the min-of-two-critics is merely pessimistic. Those mechanisms are insurance that pays off on harder, sparser, or higher-dimensional tasks. (Single seed; the gaps among the TD3 variants are small.)
