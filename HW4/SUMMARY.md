# HW4 Summary

A short note on **what** was built, **how**, **why**, **what we reached**, and **what it means**.

## What
**Planning in RL**, in two notebooks:
- `HW4_MCTS_MuZero.ipynb`: a MuZero-style agent on `CartPole` comparing **MCTS**, **naive depth search**, and **no search**.
- `HW4_dynaq.ipynb`: tabular **Dyna-Q** on `CliffWalking`, with reward shaping and prioritized sweeping.

## How
- **MuZero/MCTS**: three learned nets, namely **representation** (`obs → latent`), **dynamics** (`latent, action → next latent, reward`), and **prediction** (`latent → policy, value`). Planning is **MCTS in latent space** (PUCT selection, dynamics expansion, value-head evaluation, discounted back-up) together with a brute-force `|A|^d` depth search over the same model. Training uses **n-step value targets** from a replay buffer.
- **Dyna-Q**: one-step Q-learning on real steps plus a learned deterministic model; **n planning updates** per real step; **potential-based reward shaping** (`F = γΦ(s') − Φ(s)`); and **prioritized sweeping** (a TD-error priority queue with a predecessor map).

## Why (small touches; the implementations were already correct)
- **Dyna-Q terminal-bootstrap mask**: bootstrap only when the episode does not truly terminate (no effect on CliffWalking, but correct for terminal-rich environments such as Taxi).
- **MCTS comparison reseed**: reseed before each search type so that `mcts`, `naive`, and `none` start identically, so the comparison reflects the **search** rather than RNG luck.

## What we reached (verified by running)
- **Dyna-Q** learns the **optimal −13** CliffWalking path; more planning gives better early sample efficiency (first-30-episode mean of roughly −143 at n=0 versus −88 at n=50); reward shaping speeds up early learning; and prioritized sweeping reaches the optimum with fewer planning updates.
- **MuZero with MCTS** climbs from about **20 to about 172 out of 200** on CartPole within 100 episodes, and MCTS converges faster and more steadily than naive search, which in turn beats no search.

## What it means
- **Planning pays off**: replaying a learned model (Dyna-Q) or searching it (MCTS) turns a small number of real interactions into a strong policy, which is the central promise of model-based RL.
- **MCTS scales better than brute force**: its cost is **linear in the simulation budget**, whereas naive search is **exponential in depth**, and PUCT spends that budget adaptively on promising actions.
- **MuZero works without reconstructing observations**: its latent state only needs to be sufficient for predicting reward, value, and policy (value-equivalence).
