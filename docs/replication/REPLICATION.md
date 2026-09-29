# Replicating the paper

How the paper's method, configuration and protocol map to this repository's commands. Setup is in the
[README](../../README.md); every command runs from the repository root.

| Paper | Code |
|---|---|
| Plan refiner `F_theta` (Sec. 4, App. C.2) | `rp1.methods.rp1.net.PlannerNet`, trained by `rp1.methods.rp1.train` |
| Goal-conditioned MRN critic `V_psi` (App. C.1) | `rp1.core.agent.value.QuasimetricHead`, trained offline by `rp1.training.phases.agent.metric` and co-trained in `rp1.methods.rp1.train` |
| Frozen world-model rollout `H_phi` (Eq. 1) | `rp1.core.world_model.rollout` |
| RP1 at plan time | `rp1.methods.rp1.solver.RP1Solver` (`core/agent/solver=rp1`) |
| The RP1 configuration (Tab. 6) | the defaults of `configs/training/posttrain.yaml` and `configs/methods/rp1/` |
| Checkpoint selection (App. D.1, Tab. 5) | `pixi run select` (`rp1.inference.selection`) |
| Baselines CEM / MPPI / Adam (App. D.5) | `core/agent/solver=cem`, `mppi`, `adam` |
| Latent vs. value objective | `core/agent/value=latent` vs. `core/agent/value=metric` |
| No-op floor (Cube) | `core/agent/policy=no_move` |
| DMPO and L2O-MPC baselines | [DMPO](../dmpo/README_dmpo.md), [L2O-MPC](../l2o/README_l2o.md) |

## 1. The configuration

Every RP1 result uses one configuration across environments and world models (Tab. 6). It is the default
of `posttrain`, so a cell needs no recipe overrides. Two settings follow from the task rather than the
recipe:

- **the discount** of the offline value and the co-trained critic is set by the evaluation horizon:
  `training.gamma=0.98` (the default) for h25 and `training.gamma=0.99` for h100, so each TwoRoom and Cube
  cell is two agents, one per horizon;
- **Reacher's value reads two frames** (`training.value.window_frames=2 training.value.window_lag=5`),
  since joint velocity is not observable from one. The paper also standardizes Reacher's latents; this
  repository does not implement that step.

## 2. Training one agent

`posttrain` runs the agent pipeline against a frozen world model: latent caches at one and at `frameskip`
primitive steps per row, the action h5, the offline value, and actor-critic training of the planner.
Training uses episodes 0–7999 (`training.train_episodes=8000`).

```bash
pixi run prepare job=fetch_dataset preparation.dataset=ogb_cube
pixi run posttrain \
    training.wm=assets/core/world_model/cube_lewm \
    training.dataset=$RP1_DATA_HOME/datasets/ogb_cube_single.lance \
    training.name=cube_lewm runtime.seed=0
```

The run's `checkpoints/` holds the offline value `value_td` with a snapshot `value_td_step<N>` every
3,000 steps, and the planner `planner.pt` with a snapshot `planner_step<N>.pt` every 2,000 steps, next to
`value_ac`, the co-trained critic the planner deploys with. Caches go to `$RP1_DATA_HOME/caches/`, named
after `training.name`; give each world model its own name. A planner checkpoint records its training seed,
its step and the offline value it was trained against.

## 3. A reported cell: training grid and checkpoint selection

A reported number is not the last iterate of one run. The paper trains six seeds, each with one planner
per teacher snapshot, and selects a single (teacher snapshot, planner step) pair per cell on the selection
draws 48–51, scored as the mean over seeds and draws. Ties go to the smaller teacher budget, then the
earlier planner step. The selected pair is then evaluated on the report draws 42–44, and the cell reports
the median over the six seeds of each seed's mean. Selection draws are never used for reported numbers.

For one seed:

```bash
common="training.wm=assets/core/world_model/cube_lewm training.dataset=$RP1_DATA_HOME/datasets/ogb_cube_single.lance training.name=cube_lewm runtime.seed=$SEED"
# the caches and the offline value with its snapshots
pixi run posttrain $common "training.stages=[cache,subsample,actions,value]" logging.run_root=grid/s$SEED/teacher
# one planner per teacher snapshot
for teacher in grid/s$SEED/teacher/*/*/checkpoints/value_td*; do
    pixi run posttrain $common "training.stages=[planner]" training.teacher=$teacher \
        logging.run_root=grid/s$SEED/planner_$(basename $teacher)
done
# every planner checkpoint on every selection and report draw
for planner in grid/s$SEED/planner_*/*/*/checkpoints/planner*.pt; do
    for draw in 48 49 50 51 42 43 44; do
        pixi run evaluate benchmark=cube_lewm runtime.seed=$draw core/agent/solver=rp1 \
            core.agent.solver.checkpoint.path=$planner logging.run_root=evaluations
    done
done
```

Once every seed is evaluated:

```bash
pixi run select "selection.runs=[evaluations]"
```

`select` prints each complete pair's selection score, marks the chosen one, and writes the chosen pair,
the per-seed report means and their median to `metrics/selection.json`. A pair missing a seed or a draw
is left out.

## 4. Evaluating

Each `evaluate` run is one environment × world model × planner × objective × horizon × draw. The benchmark
sets the environment, the world model and the held-out episodes 8000–9999;
`benchmark.goal_offset_steps=25 planning.budget=50` is h25 (the default) and
`benchmark.goal_offset_steps=100 planning.budget=200` is h100. Every planner runs open loop, replanning
every 5 chunks (`planning.receding_horizon=5`). Each run writes `metrics/metrics.json`.

```bash
pixi run evaluate benchmark=cube_lewm core/agent/solver=rp1 core.agent.solver.checkpoint.path=<planner.pt>
pixi run evaluate benchmark=cube_lewm core/agent/solver=cem
pixi run evaluate benchmark=cube_lewm core/agent/solver=mppi
pixi run evaluate benchmark=cube_lewm core/agent/solver=adam
pixi run evaluate benchmark=cube_lewm core/agent/solver=cem \
    core/agent/value=metric core.agent.value.checkpoints=[<value_td>]
pixi run evaluate benchmark=cube_lewm core/agent/policy=no_move
```

The other benchmarks are `cube_pldm`, `tworoom_lewm`, `tworoom_pldm`, `reacher_lewm` and `reacher_pldm`.
On Linux GPU nodes MuJoCo renders with `MUJOCO_GL=egl` and a pinned `MUJOCO_EGL_DEVICE_ID`. Check that
the pinned device is the GPU: a node can also list a software renderer. Run one evaluation per GPU, and not
beside a training process on the same GPU, where NVIDIA EGL can abort.

The vendored agents in `assets/core/agent/` predate the configuration of Tab. 6: they are the final
iterates of an earlier recipe, without checkpoint selection. On Cube they score 86.0, 86.7 and 88.0 (LeWM
seeds 0–2) and 82.0 (PLDM seed 0) on the report draws, which makes them a check of the evaluation stack,
not a replication of the paper's numbers.

## 5. World models

The LeWM Cube world model can be trained from scratch; [cube/REPLICATION_CUBE.md](cube/REPLICATION_CUBE.md)
records the dataset fingerprint, the training command and the pitfalls. For TwoRoom and Reacher, train
on the environment's play data:

```bash
pixi run prepare job=collect_tworoom_mixed preparation.expert=0 preparation.random=10000 preparation.out=tworoom_play.lance
pixi run pretrain data=tworoom_lewm
pixi run pretrain data=reacher_lewm
```

The PLDM world models are the authors' checkpoints converted into the LeWM key layout, which shares the
architecture. `job=convert_pldm` converts another PLDM export; pair the result with the `config.json` of
`assets/core/world_model/cube_pldm`. `pixi run pretrain --config-name phases/world_model/pldm` trains
one from scratch with the authors' objective, on PushT by default (`data=` selects another dataset).

```bash
pixi run prepare job=convert_pldm preparation.src=<pldm.pt> preparation.dst=<out.pt>
```

The public Reacher h5 pads each episode's terminal step with NaN actions, so any normalization of its
actions must ignore NaNs; the trainers here do.

## 6. Dyna (App. D.4)

The Dyna-finetuned Cube world models are vendored: `benchmark=cube_lewm_dyna` and
`benchmark=cube_pldm_dyna` evaluate them with any planner, and `posttrain` with
`training.wm=assets/core/world_model/cube_lewm_dyna` retrains their agents. A Dyna round is a procedure:

1. collect episodes with the previous round's planners (all six seeds) on training-split tasks, 1,800 at
   h25 and 1,800 at h100;
2. mix them 50:50 with the offline data;
3. finetune the world model for one epoch at learning rate 1e-5 from the previous round's model
   (`pixi run pretrain training.initial_weights=<base>` on the mixture);
4. retrain RP1 on the finetuned model at the cell's selected teacher budget and planner step.

The paper's finetuning also adds a latent anchor, `||enc_ft(x) - enc_prev(x)||^2` at weight 1.0 on every
frame; `pretrain` does not implement it. The paper reports Dyna at h25, round 1 for LeWM and round 2 for
PLDM.
