"""Render the environment illustration figure used in the paper.

Renders one frame per evaluation environment straight from the registered
Stable-WM envs and lays them out as a labelled panel.  Because every task in
rp1 is goal-conditioned, each panel also marks where the goal is; ``--style``
picks how.

    ghost  translucent copy of the goal state drawn into the frame
    star   gold star marker at the goal location
    inset  goal frame as a thumbnail in the panel corner

    python scripts/make_env_figure.py --style ghost

The OGBench arm is re-shaded (light UR5e materials, shadows off) purely so it
reads on a printed page -- the geometry, camera and cube layout are untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import stable_worldmodel  # noqa: F401  (registers the swm/* envs)
from matplotlib.patches import Circle

RES = 512
GOAL_COLOR = "#f5c518"
GOAL_EDGE = "#2b2b2b"
RING_COLOR = "#1faa4b"

# `front_pixels` is what get_pixel_observation() renders, so it is the view the
# policy sees and the one the evaluation videos are recorded from.
VIDEO_CAMERA = "front_pixels"
# Arm start that keeps the wrist and gripper inside that camera's tight crop.
EE_START = np.array([0.58, 0.0, 0.20], dtype=np.float32)

# Light UR5e materials so the arm is legible against the dark table.
ARM_MATERIALS = {
    "ur5e/black": (0.35, 0.36, 0.38),
    "ur5e/jointgray": (0.62, 0.63, 0.66),
    "ur5e/linkgray": (0.86, 0.87, 0.89),
    "ur5e/lightblue": (0.45, 0.62, 0.82),
    "ur5e/robotiq/metal": (0.70, 0.71, 0.74),
    "ur5e/robotiq/silicone": (0.25, 0.26, 0.28),
    "ur5e/robotiq/gray": (0.62, 0.63, 0.66),
    "ur5e/robotiq/black": (0.32, 0.33, 0.35),
    "ur5e/robotiq/pad_gray": (0.55, 0.56, 0.58),
}


def project(point, cam_pos, cam_mat, fovy_deg, width, height):
    """Project a world point into pixel coordinates for a MuJoCo camera."""
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    cam = cam_mat.T @ (np.asarray(point, dtype=float) - cam_pos)
    depth = -cam[2]
    return width / 2.0 + focal * cam[0] / depth, height / 2.0 - focal * cam[1] / depth


def blend(base, overlay, mask, alpha):
    out = base.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * overlay[mask].astype(np.float32)
    return out.astype(np.uint8)


# --------------------------------------------------------------------------- reacher


def make_reacher(seed=0, goal_seed=7):
    """Start pose plus a goal joint configuration (the qpos_match task)."""
    env = gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array")

    env.reset(seed=goal_seed)
    goal_qpos = env.unwrapped.env.physics.data.qpos.copy()

    env.reset(seed=seed)
    physics = env.unwrapped.env.physics
    frame = np.asarray(env.render())

    with physics.reset_context():
        physics.data.qpos[:] = goal_qpos
    goal_frame = np.asarray(env.render())
    goal_finger = physics.named.data.geom_xpos["finger"].copy()
    cam = physics.model.name2id("fixed", "camera")
    goal_px = project(
        goal_finger,
        physics.data.cam_xpos[cam].copy(),
        physics.data.cam_xmat[cam].reshape(3, 3).copy(),
        physics.model.cam_fovy[cam],
        frame.shape[1],
        frame.shape[0],
    )

    env.close()

    # The arm is warm-toned, the backdrop is blue -- a channel test isolates it.
    arm = goal_frame[..., 0].astype(int) > goal_frame[..., 2].astype(int) + 20
    return {
        "frame": frame,
        "goal_frame": goal_frame,
        "goal_px": goal_px,
        "ghost": blend(frame, goal_frame, arm, 0.55),
    }


# --------------------------------------------------------------------------- cube


def make_cube(seed=2):
    env = gym.make("swm/OGBCube-v0", render_mode="rgb_array", ob_type="pixels", width=RES, height=RES)
    # The arm start is a reset variation, so it has to be supplied through reset
    # options -- assigning to the space is undone by the reset itself.  This pose
    # brings the wrist and gripper down into the camera's crop.
    env.reset(seed=seed, options={"variation_values": {"agent.ee_start_position": EE_START}})
    unwrapped = env.unwrapped
    model, data = unwrapped._model, unwrapped._data

    for name, rgb in ARM_MATERIALS.items():
        mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if mat >= 0:
            model.mat_rgba[mat, :3] = rgb
            model.mat_rgba[mat, 3] = 1.0
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.85
    model.light_castshadow[:] = 0
    model.light_diffuse[:] = 0.75

    block = unwrapped._target_block
    for gid in unwrapped._cube_target_geom_ids_list[block]:
        model.geom(gid).rgba[3] = 0.35

    goal_pos = data.mocap_pos[unwrapped._cube_target_mocap_ids[block]].copy()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, VIDEO_CAMERA)
    goal_px = project(
        goal_pos,
        data.cam_xpos[cam].copy(),
        data.cam_xmat[cam].reshape(3, 3).copy(),
        model.cam_fovy[cam],
        RES,
        RES,
    )
    frame = np.asarray(unwrapped.render(camera=VIDEO_CAMERA))

    # Goal frame: the cube itself sitting on the target pose.
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_0")
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    saved = data.qpos[adr : adr + 3].copy()
    data.qpos[adr : adr + 3] = goal_pos
    mujoco.mj_forward(model, data)
    goal_frame = np.asarray(unwrapped.render(camera=VIDEO_CAMERA))
    data.qpos[adr : adr + 3] = saved
    mujoco.mj_forward(model, data)
    env.close()

    return {"frame": frame, "goal_frame": goal_frame, "goal_px": goal_px, "ghost": frame}


# --------------------------------------------------------------------------- tworoom


def make_tworoom(seed=7):
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    frame = np.asarray(env.render())
    target = unwrapped.target_position.cpu().numpy()
    goal_frame = unwrapped._render_frame(agent_pos=unwrapped.target_position).cpu().numpy().transpose(1, 2, 0)

    radius = float(unwrapped.variation_space["agent"]["radius"].value.item())
    env.close()

    # A translucent dot would read as a second agent here, so TwoRoom marks its
    # goal with an open ring instead.
    return {
        "frame": frame,
        "goal_frame": goal_frame,
        "goal_px": (float(target[0]), float(target[1])),
        "ghost": frame,
        "ring": 1.9 * radius,
    }


# --------------------------------------------------------------------------- figure

PANELS = (
    ("reacher", "Reacher", make_reacher),
    ("cube", "OGBench Cube", make_cube),
    ("tworoom", "TwoRoom", make_tworoom),
)


def decorate(ax, panel, style: str) -> None:
    """Draw one env frame plus its goal annotation into an axes."""
    ax.imshow(panel["ghost"] if style == "ghost" else panel["frame"])

    if style == "ghost" and "ring" in panel:
        ax.add_patch(Circle(panel["goal_px"], panel["ring"], fill=False, edgecolor=RING_COLOR, linewidth=2.6))

    if style == "star":
        x, y = panel["goal_px"]
        ax.plot(x, y, marker="*", markersize=26, color=GOAL_COLOR, markeredgecolor=GOAL_EDGE, markeredgewidth=1.1)

    if style == "inset":
        inset = ax.inset_axes([0.66, 0.66, 0.32, 0.32])
        inset.imshow(panel["goal_frame"])
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor(GOAL_COLOR)
            spine.set_linewidth(1.8)
        inset.set_xlabel("goal", fontsize=8, family="serif", labelpad=2)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.6")
        spine.set_linewidth(0.8)


def draw(panels, style: str, out: Path, dpi: int, wspace: float = 0.06) -> None:
    """One combined strip, labels baked in.

    A wider ``wspace`` makes the strip wider without making the panels taller,
    so at a fixed ``\\linewidth`` the figure renders shorter on the page.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
    for ax, (_, label, panel) in zip(axes, panels, strict=True):
        decorate(ax, panel, style)
        ax.set_title(label, fontsize=13, family="serif", y=-0.14)

    fig.subplots_adjust(wspace=wspace, left=0.01, right=0.99, top=0.99, bottom=0.09)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def draw_split(panels, style: str, outdir: Path, dpi: int) -> None:
    """One file per panel, unlabelled, for \\subcaptionbox in LaTeX."""
    outdir.mkdir(parents=True, exist_ok=True)
    for slug, _, panel in panels:
        fig, ax = plt.subplots(figsize=(3, 3))
        decorate(ax, panel, style)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        out = outdir / f"env_{slug}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.01, facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=("ghost", "star", "inset", "all"), default="all")
    parser.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--split",
        action="store_true",
        help="also write one unlabelled file per panel, for LaTeX subfigures",
    )
    parser.add_argument("--wspace", type=float, default=0.45, help="gap between panels, in panel widths")
    args = parser.parse_args()

    # Render the scenes once: the envs resample some variations on every
    # construction, so rebuilding per style would give each option a different
    # layout and make them impossible to compare.
    panels = [(slug, label, fn()) for slug, label, fn in PANELS]

    styles = ("ghost", "star", "inset") if args.style == "all" else (args.style,)
    for style in styles:
        draw(panels, style, args.outdir / f"environments_{style}.png", args.dpi, args.wspace)
        if args.split:
            draw_split(panels, style, args.outdir / f"panels_{style}", args.dpi)


if __name__ == "__main__":
    main()
