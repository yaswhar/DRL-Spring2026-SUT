# HW3 — Implementation Instructions for Claude Code

These are the coding tasks for HW3, all inside **`HW3_DPGs.ipynb`** with the environment defined in **`HW3_moving_target_reacher_env.py`**. Theory is solved in `HW3-Solution.md`; do **not** touch that. Fill in every `# TODO` / `...` in the notebook and run all cells top-to-bottom. Environment: `gymnasium[mujoco]`, a moving-target `Reacher-v5` variant; observations are flat vectors, continuous bounded actions.

Build the three agents in order — **DPG → DDPG → TD3** — because each extends the previous. Keep the shared `train_agent`, `evaluate`, replay buffer, and plotting utilities reusable across all three.

## Shared utilities (cells 14–17)
- `soft_update(net, target_net, tau)`: in-place Polyak update `θ_targ ← (1−τ)θ_targ + τθ` over `zip(net.parameters(), target_net.parameters())`, under `@torch.no_grad()`.
- `DeterministicActor`: MLP trunk, output squashed to the action box with `a = b + scale·tanh(f_θ(o))`, where `scale = (a_high−a_low)/2`, `b = (a_high+a_low)/2`. Register `scale`/`b` as buffers.
- `Critic`: MLP on `concat([obs, act])` → scalar `Q`.
- `train_agent(...)`: standard env loop — select action (policy + exploration noise, clipped to action box), step, store transition, update agent, log episodic return; return the learning-curve history.
- `evaluate(env, agent, episodes, seed)`: run the **deterministic** policy (no exploration noise), return mean/return list.

## Implementation 1 — DPG (cells 13–20) [5 pts]
Minimal near-on-policy deterministic actor-critic.
- Critic target (one-step, optionally with target nets): `y = r + γ(1−d)·Q(o', μ(o'))`.
- Critic loss: `L_Q = (Q(o,a) − y)²` (MSE over the batch / minibatch).
- Actor loss: `L_μ = −Q(o, μ(o))` (ascend the critic value via the DPG direction `∇_θ μ_θ(s)∇_a Q`).
- Collect data with noisy actions `a = μ(o) + ε`, `ε ~ N(0, σ²)`, clipped to the action range.
- Train, plot the learning curve, and in the report **discuss why this baseline is less stable / less sample-efficient than DDPG and TD3** (near-on-policy correlated data, moving bootstrap target, no replay reuse).

## Implementation 2 — DDPG (cells 21–26) [5 pts]
Extend DPG to off-policy DDPG.
- Add a **replay buffer**; sample random minibatches `(s,a,r,s',d) ~ D`.
- Add **target networks** `Q_φ̄`, `μ_θ̄`, soft-updated with `τ` each step.
- Critic target: `y = r + γ(1−d)·Q_φ̄(s', μ_θ̄(s'))`; critic loss `L_Q = E[(Q_φ(s,a) − y)²]`.
- Actor objective: `J(θ) = E_{s∼D}[Q_φ(s, μ_θ(s))]`, i.e. minimize `L_π = −E[Q_φ(s, μ_θ(s))]`.
- Collect with exploration noise; **evaluate without noise**.
- Plot DDPG vs DPG; in the report explain how the replay buffer (sample efficiency + decorrelation) and target networks (stationary bootstrap target) fix DPG's instabilities.

## Implementation 3 — TD3 (cells 27–34) [10 pts]
Extend DDPG with the three TD3 mechanisms. Add to `TD3Config`: `policy_delay` (e.g. 2), `target_noise_std` (e.g. 0.2), `target_noise_clip` (e.g. 0.5). In `TD3Agent`: create a **second critic** `Q2` and its target, hard-init both targets, and use a single optimizer over both critics' params.
1. **Clipped double-Q**: target `y = r + γ(1−d)·min_{i=1,2} Q_{i,targ}(s', a')`.
2. **Target policy smoothing**: `a' = clip(μ_targ(s') + ε, a_min, a_max)`, `ε ~ clip(N(0,σ²), −c, c)`.
3. **Delayed updates**: update actor and all target nets once every `policy_delay` critic updates. Delayed actor loss `L_π = −E[Q1(s, μ_θ(s))]`.
- Critic loss: `MSE(Q1(s,a), y) + MSE(Q2(s,a), y)`.

### Ablations (cell 34) — required
Train three single-change ablations, each removing exactly one TD3 addition: (a) no clipped double-Q (single critic), (b) no target smoothing, (c) no delayed updates (actor every critic step). Plot learning curves for **DPG, DDPG, full TD3, and the three ablations** together. In the report, state which component mattered most for performance/stability and connect it to critic **overestimation** and the actor **exploiting critic errors** (tie back to Q7–Q14 of the theory).

## Deliverables
Completed `HW3_DPGs.ipynb` (all cells run), learning-curve plots, evaluation rollout video(s), and a short written discussion for each implementation. Use fixed seeds and report mean ± std over a few eval episodes.
