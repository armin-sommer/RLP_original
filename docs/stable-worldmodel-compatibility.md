# Stable-WM compatibility

rp1 depends on the published `stable-worldmodel==0.1.1`, pinned because its adapters target that
release's module layout and runtime contracts. What rp1 needs beyond the release lives in small
adapters next to the code that uses them:

- `rp1.environment.World` adds dataset-backed resets, goal snapshots, recording and image resizing.
- `rp1.core.agent.value.LatentGoalCost` fixes the candidate-axis broadcasting of the LeWM/PLDM terminal
  cost.
- `rp1.core.agent.policy.WorldModelPolicy` keeps real observation and action history at the training
  cadence, where the release pads a single observation.
- `rp1.training.harness.checkpointing` resolves checkout-relative checkpoint paths and accepts plain
  mappings as saved configs.
- `rp1.core.agent.solver.GradientSolver` moves full-horizon warm starts to the solver's device.

## Upgrading past 0.1.1

Upstream `main` is not a drop-in upgrade:

- solver modules moved under `stable_worldmodel.planning`, and their constructors take a cost object
  rather than a model;
- `World` step callbacks and dataset reset extraction changed their shapes and masking;
- LeWM predictions became future-only, which changes action-history semantics;
- checkpoint saving no longer needs the mapping workaround; and
- its training stack requires `stable-pretraining>=0.1.8`, while the vendored checkpoints are validated
  with 0.1.7.

An upgrade is a migration: port the planning adapters, revalidate dataset resets and callbacks, migrate
checkpoint loading, and rerun the latent and rollout parity checks before moving the pin.
