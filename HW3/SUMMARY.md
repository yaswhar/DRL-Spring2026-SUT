# HW3 & HW4 — Summary

A short note on **what** was built, **how**, **why**, **what we reached**, and **what it means**.

## What
- **HW3 — Deterministic actor–critic** on a moving-target MuJoCo `Reacher`: build **DPG → DDPG → TD3** and ablate TD3's three additions.
- **HW4 — Planning**: a MuZero-style agent doing **MCTS vs. naive depth search** on `CartPole`, and tabular **Dyna-Q** (with reward shaping and prioritized sweeping) on `CliffWalking`.

## How
- **HW3**: deterministic `tanh`-squashed actor + Q-critic; one-step DPG; then DDPG adds a **replay buffer + target networks + OU exploration**; then TD3 adds **clipped double-Q**, **target-policy smoothing**, and **delayed actor/target updates**. Bootstrapping masks true termination only.
- **HW4**: learned **representation / dynamics / prediction** nets; full **MCTS** (PUCT selection, dynamics expansion, value backup) and brute-force depth search over the learned model; **n-step value targets**; tabular Dyna-Q with a learned model, **potential-based** reward shaping, and **prioritized sweeping**.

## Why (two real fixes in HW3)
Careful tracing — not just "it runs" — found two pitfalls in the continuous actor–critic:
1. **Actor `tanh` saturation → dead policy gradient.** Large pre-activations rail `tanh` to ±1, the gradient vanishes, and training goes **flat**. Fixed with a **small final-layer init** (`±3e-3`) so initial actions start near 0.
2. **Uncontrolled per-agent randomness.** `train_agent` reseeded only the env, so comparisons measured init luck. Fixed with **`set_global_seed()` before each agent**.
(HW4 was already correct; added a terminal-bootstrap mask and a per-search-type reseed for cleanliness.)

## What we reached (real training)
**HW3** (25k steps/agent, seeded) — see `figures/ablation_curves.png`:

| Agent | Eval return | Fingertip error |
|---|---|---|
| DPG | −25.8 | ~15 cm (never tracks) |
| DDPG | −1.99 | ~1.3 cm |
| **TD3 (full)** | **−1.72** | **~1.1 cm (best)** |

Ablations (train return, last 20): full **−2.82**; no-smoothing **−3.45**; no-delay −2.89; no-double-Q −2.86.
