# HW4 — Implementation Instructions for Claude Code

Coding tasks for HW4. Theory is solved in `HW4-Solution.md`; do **not** modify it. Fill every `# TODO` and run notebooks top-to-bottom.

---

## A. MCTS + MuZero — `HW4_MCTS_MuZero.ipynb` [20 pts]

A small MuZero-style agent on **CartPole**, comparing **MCTS** vs **naive depth search**. Cells marked *(provided)* need no edits; implement the *(TODO)* ones.

**3️⃣ Replay buffer (cell 8).** Implement the **n-step return** inside `_sample_one`: `G = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n · bootstrap_value(s_{t+n})`, with correct truncation at episode end and proper handling of the value target used by MuZero.

**4️⃣ Networks (cells 9–10).** Three small MLPs (CartPole is low-dim):
- `RepresentationNet`: observation → latent state `s⁰`.
- `DynamicsNet`: `(s^k, a^k)` → `(s^{k+1}, reward^k)`.
- `PredictionNet`: `s^k` → `(policy logits, value)`.
Keep hidden sizes small; latent dimension modest.

**5️⃣ Search (cells 15–18).**
- `MCTS` (TODO): implement the four phases — **selection** (PUCT: `argmax_a Q(s,a) + c_puct·P(s,a)·√N(s)/(1+N(s,a))`, with `MinMaxStats` normalizing Q), **expansion** (call dynamics + prediction to add a child), **simulation/evaluation** (use predicted value, no environment rollout — this is MuZero), **backpropagation** (propagate the discounted value up the path, updating `N`, `Q`). Add Dirichlet noise to the **root** prior for exploration.
- `Naive depth search` (TODO): enumerate all `|A|^d` action sequences via the learned model, score each by cumulative discounted predicted reward + terminal value, return the first action of the best sequence.

**6️⃣ Agent (cell 20).** `MuZeroAgent`: `initial_inference(obs)` = representation→prediction; `recurrent_inference(s,a)` = dynamics→prediction; expose an `act` that runs the chosen planner.

**7️⃣–8️⃣ Training/comparison (provided).** Uses your code; trains three agents and plots learning curves.

**Discussion (cell 25).** Answer: which search converges fastest and why; wall-clock cost of MCTS vs naive search; how planning depth/budget affects each; relate to the theory (`HW4-Solution.md` MCTS Q1–Q2: misleading shallow estimates, prior bias, when naive depth search wins).

---

## B. Dyna-Q — `HW4_dynaq.ipynb` (if assigned)

Tabular Dyna-Q on **Cliff Walking**. Implement the `# TODO` cells:
- `greedy_policy` / `epsilon_greedy_policy`.
- `q_planning(model, q, alpha, gamma, n)`: `n` planning steps sampling visited `(s,a)` pairs uniformly, applying the Q-learning update from the model's stored `(r, s')`.
- `dyna_q(...)`: per real step — ε-greedy action, env step, direct Q-learning update, store in model, then `n` planning steps.
- **Experiments**: sweep `n ∈ {0, 5, 50}` (n=0 ≡ Q-learning), plot rolling-mean reward curves on one axis.
- **Improvement** section: implement and evaluate one enhancement (e.g., decaying ε/α).
- **Reward shaping**: use `ShapedCliffWalkingEnv`; compare early vs late learning.
- **Prioritized sweeping**: build a `predecessors` map; `q_planning_priority` with a priority queue keyed by TD-error magnitude (threshold θ); `dyna_q_priority`; compare to vanilla Dyna-Q at equal planning-update budget.
- Answer the embedded discussion questions; optional bonus (stochastic Dyna-Q etc.).

---

## Deliverables
Both notebooks fully run, all learning-curve plots produced, evaluation videos where the scaffold creates them, and written answers to every discussion question. Use the provided seeds for reproducibility.
