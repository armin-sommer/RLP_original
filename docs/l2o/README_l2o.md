# L2O-MPC — Learning to Optimize in MPC as an rp1 baseline

Sacks, Boots, *Learning to Optimize in Model Predictive Control*, ICRA 2022
([arXiv:2212.02603](https://arxiv.org/abs/2212.02603), the direct predecessor
of the DMPO baseline — same first author).

The paper unifies sampling-based MPC under dynamic mirror descent and learns
the **whole update rule**: a two-layer MLP reads the current sampling
distribution and the `N` rollout costs and emits a sigmoid-gated replacement

    mu = (1 - g) . mu~  +  g . h,        [g, h] = m_theta([costs, mu~, sigma])

where `mu~` is the warm-started (shifted) mean. That is the structural
contrast with DMPO, which keeps the hand-written MPPI reduction and learns a
*residual* on it: L2O-MPC's network never sees an MPPI mean — untrained, it is
not a working optimizer at all, which is why its training is imitation.

**Training is DAgger against a many-sample MPPI expert** — the same optimizer
with a larger sample budget (`N` expert vs `M <= N` learner; the paper's
headline is matching a 512-sample optimizer with 16). The learner's update is
regressed (MSE) onto the expert's, with `beta_k = 0.8^k` state mixing over 20
rounds. This is the one baseline whose training method transfers to this
repository *without modification*: the expert is exactly computable here (the
hand-written MPPI update through the frozen world model and frozen critic), so
unlike DMPO (PPO → pathwise) no gradient-method substitution is needed.

## Paper component → code

| Paper | Code |
|---|---|
| Gated update rule `m_theta` (GRU-style, sigmoid gates) | `L2ONet.forward` — `rp1.core.agent.planner.l2o` |
| DMD-MPC / MPPI expert update (Eq. 12) | `mppi_update` (free function, arbitrary sample count) |
| Fixed Halton sample set | `L2ONet.plans` via `gaussian_halton` (shared with DMPO) |
| Standard shift warm start (no learned shift) | `L2ONet.warm_start` |
| DAgger training, `beta_k = 0.8^k` over 20 rounds | `rp1.training.phases.agent.l2o.dagger_beta` |
| Fixed diagonal covariance (paper experiments) | `learn_std=false` default; the paper's gated covariance formulation behind `learn_std=true` |
| Deployment | `L2OSolver` — `rp1.core.agent.solver.l2o` |

Paper hyperparameters carried over: two ReLU hidden layers with dropout 0.1,
Adam at 1e-3, 20 DAgger rounds with decay 0.8, Halton samples, fixed diagonal
covariance, expert `N` ≫ learner `M`.

## What differs from the paper, and why

1. **Dynamics inside the optimizer are the frozen learned world model**, not
   the simulator's ground-truth dynamics. Every planner row in this repository
   shares that substitution; it is what makes the rows comparable, and it means
   L2O-MPC here inherits world-model error the paper's controllers never saw.
2. **Plan costs are the goal-conditioned quasimetric critic**, not a
   hand-designed task cost — the same critic the rp1, DMPO, and value-CEM/MPPI
   rows plan against, so the table isolates the planner.
3. **DAgger's state distribution is the inner loop, not the receding-horizon
   loop.** The paper visits states by running the controller closed loop in
   the sim, warm-starting each step. This repository's protocol is open loop
   (a full 5-block plan executes per decision, cold start), so the sequential
   state the learner is trained and deployed on is the chain of `K` inner-loop
   iterates on offline cache windows. The expert relabels every iterate; the
   `beta` coin decides whether the chain advances along expert or learner
   updates.
4. **Cost features are standardized** before entering the network (the sibling
   DMPO reference code conditions its cost features the same way); the paper
   does not specify its cost conditioning.
5. **Action bounds are the symmetric plan clip** `[-amax, amax]` shared with
   the rest of the repository, matched per cell to the other planner rows.
6. **No real-system interaction at training time.** Nothing here (or in the
   paper) optimizes real return — the paper's training signal is also
   model-rollout cost under its simulator; real performance appears only at
   evaluation.

## Cost per decision

| planner | world-model unrolls per decision |
|---|---|
| CEM / MPPI (repo defaults) | 9,000 forward (300 samples × 30 iterations) |
| Adam | 3,000 forward + 3,000 backward |
| DMPO (defaults) | 256 forward (256 × 1) |
| **L2O-MPC** (defaults) | **256 forward (64 × 4)** — DMPO's exact budget |
| rp1 / rp1 | 9 forward + 8 backward |

The sample count is baked into the trained network (costs are read
positionally); the iteration count can be varied at test time
(`core.agent.solver.iters`), as in the paper's sweeps.

## Commands

L2O-MPC trains against a frozen value, so first run the agent pipeline without its planner stage to
produce the caches and `value_td`:

```bash
pixi run posttrain training.wm=assets/core/world_model/cube_lewm \
    training.dataset=$RP1_DATA_HOME/datasets/ogb_cube_single.lance training.name=cube_lewm \
    "training.stages=[cache,subsample,actions,value]"
```

Then train and evaluate:

```bash
pixi run posttrain --config-name phases/agent/l2o training.wm=assets/core/world_model/cube_lewm \
    training.cache=$RP1_DATA_HOME/caches/cube_lewm_fs5.pt \
    training.h5=$RP1_DATA_HOME/caches/cube_lewm_actions.h5 \
    training.init_value=<run>/checkpoints/value_td core.agent.planner.action_limit=1.6
pixi run evaluate benchmark=cube_lewm core/agent/solver=l2o core.agent.solver.checkpoint.path=<l2o.pt>
```

The hand-written counterpart, what the expert computes deployed at the standard budget, is MPPI under
the same value:

```bash
pixi run evaluate benchmark=cube_lewm core/agent/solver=mppi \
    core/agent/value=metric core.agent.value.checkpoints=[<value_td>]
```
