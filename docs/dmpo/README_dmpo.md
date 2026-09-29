# DMPO — Deep Model Predictive Optimization as an rp1 baseline

Sacks, Rana, Huang, Spitzer, Shi, Boots, *Deep Model Predictive
Optimization*, ICRA 2024 ([arXiv:2310.04590](https://arxiv.org/abs/2310.04590),
code [`jisacks/dmpo`](https://github.com/jisacks/dmpo)).

DMPO and rp1 answer the same question — *replace the hand-designed planner with
a learned one* — from opposite ends. rp1 learns a **refiner** that moves a plan
along the value gradient (9 world-model rollouts per decision). DMPO learns the
**reduction inside a sampling optimizer**: keep MPPI's sample-rollout-reduce
loop, and let an MLP turn the `N` rollout costs into the next sampling
distribution. It is the strongest available "learned optimizer" baseline for
the rp1 tables, and it is a *residual* on MPPI, so the comparison is clean:
untrained, it reproduces the MPPI row exactly.

## Paper component → code

| Paper | Code |
|---|---|
| Learned update rule `m_phi` (Eq. 13–14: gated MPPI residual, multiplicative covariance) | `DMPONet.forward` — `rp1.core.agent.planner.dmpo` |
| Learned warm start / shift model `Phi_phi` (Sec. IV-D) | `DMPONet.warm_start` |
| Fixed Halton sample set, current mean always sampled | `gaussian_halton`, `DMPONet.plans` |
| MPPI inner update (Eq. 5–6, min-max cost scaling, dynamic mirror descent step) | `DMPONet.mppi_mean` |
| `is_mppi` ablation | `core.agent.solver.mppi_mode=true` |
| Deployment (rollouts + cost + inner loop) | `DMPOSolver` — `rp1.core.agent.solver.dmpo` |
| Training | `rp1.training.phases.agent.dmpo`, config `configs/training/phases/agent/dmpo.yaml` |

Reference hyperparameters (one 256-unit ReLU hidden layer, last layer
`N(0, 1e-3)`, temperature 0.05, step size 0.8, cost scaling on, gate, learned
covariance, one inner iteration) are the defaults in
`configs/core/agent/planner/dmpo.yaml`.

## Two trainers: `phases/agent/dmpo` and `phases/agent/dmpo_ppo`

| | `phases/agent/dmpo` (pathwise) | `phases/agent/dmpo_ppo` (offline DMPO) |
|---|---|---|
| objective | `V(z_T(mu_K), z_g)` of **one** decision | discounted return over `decisions` closed-loop decisions |
| algorithm | backprop through the frozen world model | PPO + GAE, forward-only rollouts |
| search heads | absent (deterministic update) | present — the sampled `(mu, Sigma)` is the policy action |
| critic | — | `DMPOCritic` over `(z_t, z_g, theta_{t-1})`, the paper's auxiliary state |
| trains the shift model | no (single decision) | yes (credit crosses decisions) |
| env steps | 0 | 0 |

`phases/agent/dmpo_ppo` is the paper's *algorithm*, closed inside the world model:
each imagined episode runs the whole MPC-in-the-loop policy for several
decisions, the reward is progress in the critic's cost-to-go
`V(z_t, z_g) - V(z_{t+1}, z_g)`, and PPO optimizes the discounted sum. Nothing
differentiates through the world model, exactly as on hardware. It is the
closer reproduction and the one to prefer when the DMPO row has to defend
itself as DMPO; `phases/agent/dmpo` remains the cheaper apples-to-apples comparison
against rp1, which is trained pathwise in the same way.

Both write the same checkpoint format, so `core/solver=dmpo` deploys either
(deployment always uses the distribution *locations* — the reference's
`use_mean`).

What neither trainer reproduces: the paper measures return on the **system**,
with the model only inside the inner loop, under domain randomization. That is
what lets its learned optimizer compensate for model error — the robustness
claim. Here training and the inner loop share one frozen world model, so model
error is invisible to training and only surfaces at evaluation. Closing that
gap needs environment rollouts.

## What differs from the paper, and why

1. **Pathwise gradients instead of PPO** (`phases/agent/dmpo`; `phases/agent/dmpo_ppo` closes
   this gap for the algorithm, though not for the on-system objective). DMPO is trained with PPO because its
   costs come from a real quadrotor: no analytic gradient exists. Here the
   world model is differentiable, so the same networks are trained by
   backpropagating `V(z_T(mu_K), z_g)` through the frozen world model — the
   convention this repository's own planner uses. Consequence: the actor's
   stochastic search heads (`mean_search_std`, `std_search_std`), which exist
   only to give PPO a policy gradient, are dropped. Everything on the forward
   path is the reference computation. This makes the DMPO row *comparable* to
   the rp1 row (same data, same frozen critic, same objective, different
   learned planning procedure) but it is **not** a replication of the paper's
   quadrotor result.
2. **The gate is `tanh`, following the authors' code.** The paper's text
   describes a sigmoid gate in `[0, 1]`; `dmpo_policy.py` uses `tanh`.
   `core.agent.planner.gate_activation=sigmoid` gives the paper-literal variant.
3. **Costs are the goal-conditioned critic, not a task cost.** DMPO plans
   against the same quasimetric value the rp1 solver plans against (recorded in
   its checkpoint), so a DMPO-vs-rp1 table isolates the planner. `amax` (the
   symmetric plan clip in z-scored action units) replaces the quadrotor's
   asymmetric thrust bounds.
4. **The learned warm start is inert under the shipped eval protocol.** All
   environments run open loop (`planning.receding_horizon=5` with a 5-block
   plan), so nothing survives the shift-forward and each decision cold-starts.
   Evaluate with `planning.receding_horizon=1` for the closed-loop regime DMPO
   was published in; train the shift model with `warm_start_every>0`.

## Cost per decision

| planner | world-model unrolls per decision |
|---|---|
| CEM / MPPI (repo defaults) | 9,000 forward (300 samples × 30 iterations) |
| Adam | 3,000 forward + 3,000 backward |
| **DMPO** (defaults) | **256 forward** (256 samples × 1 iteration), no backward |
| rp1 / rp1 | 9 forward + 8 backward |

The sample count is baked into the trained network — the actor reads the `N`
costs positionally — so it cannot be changed after training; the solver logs
and ignores a mismatching config value. The iteration count can be varied at
test time (`core.agent.solver.iters`), as the paper does.

## Commands

DMPO trains against a frozen value, so first run the agent pipeline without its planner stage to
produce the caches and `value_td`:

```bash
pixi run posttrain training.wm=assets/core/world_model/cube_lewm \
    training.dataset=$RP1_DATA_HOME/datasets/ogb_cube_single.lance training.name=cube_lewm \
    "training.stages=[cache,subsample,actions,value]"
```

Then train and evaluate:

```bash
pixi run posttrain --config-name phases/agent/dmpo training.wm=assets/core/world_model/cube_lewm \
    training.cache=$RP1_DATA_HOME/caches/cube_lewm_fs5.pt \
    training.h5=$RP1_DATA_HOME/caches/cube_lewm_actions.h5 \
    training.init_value=<run>/checkpoints/value_td core.agent.planner.action_limit=1.6
pixi run evaluate benchmark=cube_lewm core/agent/solver=dmpo core.agent.solver.checkpoint.path=<dmpo.pt>
```

Offline DMPO (the PPO objective) trains with `--config-name phases/agent/dmpo_ppo` and the same
arguments. The hand-written update DMPO learns a residual on, under the same value:

```bash
pixi run evaluate benchmark=cube_lewm core/agent/solver=mppi \
    core/agent/value=metric core.agent.value.checkpoints=[<value_td>]
pixi run evaluate benchmark=cube_lewm core/agent/solver=dmpo \
    core.agent.solver.checkpoint.path=<dmpo.pt> core.agent.solver.mppi_mode=true
```

Window values (Reacher's three-frame quasimetric) work on both sides: trainer and solver read the window
width off the value's `latent_dim` and score the last imagined frames the way `MetricCost` does at
evaluation (`rp1.core.agent.value.temporal.windowed_terminal_value`).

Reporting follows the repository's protocol: selection on evaluation seeds 50/51, reports on 42/43/44
with 50 episodes each, and three training seeds per cell.
