# The Cube LeWM world model from scratch

`assets/core/world_model/cube_lewm` is a LeWM (ViT-tiny encoder, trained from scratch) on OGBench Cube.
Under the CEM evaluation below it reaches 84.0% success over 50 episodes, against 42.0% for a random
policy under the same protocol and seed.

| File | Contents |
|---|---|
| `training_config.yaml` | the resolved training and evaluation settings |
| `model_config.json` | the instantiable model config saved with the checkpoint |
| `cube_fingerprint.json` | dataset fingerprint: rows, episodes, pixel format, action statistics |
| `cube_frame_ep0_step0.png`, `cube_frame_ep0_step50.png` | raw dataset frames, the exact pixels the model trains on |

## Dataset

`quentinll/lewm-cube` on Hugging Face, the DINO-WM Cube dataset: 224×224 RGB frames (stored as JPEG in
lance) and 5-dimensional actions. Compare your renders with the two frames above; a mismatch in camera
pose, arm colors, lighting or shadows means a different environment version than Stable-WM's
`swm/OGBCube-v0`.

## Pitfalls

**Action normalization.** Evaluation fits a `StandardScaler` on the dataset's action column and
`WorldModelPolicy` applies its `inverse_transform` to every planned action before the environment sees
it. The world model must therefore be trained on actions z-scored with statistics over the full action
column; trained on raw actions, the planner's actions are off by the per-dimension standard deviation.

**NaN actions.** Episode boundaries carry NaN action rows. Training replaces them with zeros before the
statistics are computed; evaluation drops those rows instead, and the two statistics agree closely.

**Non-finite gradients.** Rare batches produce non-finite gradients. `clip_grad_norm_` then scales every
gradient by NaN and Adam never recovers, while the loss curve can look plausible for a long time. Skip
the optimizer step whenever the returned gradient norm is not finite (`NonFiniteGradientGuard`).

**Training length.** Success keeps rising until the end of the 22 epochs; a run of 5–10 epochs lands
well below the final number.

**The success floor.** With `env_type=single`, success is the cube within 4 cm of its target. Expert
episodes contain long stretches where the block does not move, so many goals 25 steps ahead are already
reached at the start; always report the random-policy floor beside a result. The evaluation seed fixes
which (episode, start) pairs are drawn.

**Precision.** Training runs under bf16 autocast and evaluation with `runtime.bfloat16=true`; keep the
two matched to reproduce the number exactly.

**Architecture.** The details that are easy to get wrong: patch size 14 (not 16), an AdaLN-conditioned
predictor, and one action token per step made of the flattened frameskip × action-dimension chunk. The
full settings are in `training_config.yaml` and `model_config.json`.

## Evaluation

```bash
pixi run evaluate benchmark=cube_lewm \
    core.world_model.checkpoint=<checkpoint> \
    data.path=<the training dataset as lance> \
    runtime.bfloat16=true
```
