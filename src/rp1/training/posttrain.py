"""Train the agent on a frozen world model: ``pixi run posttrain``.

The default config runs the full pipeline; ``--config-name phases/agent/<phase>``
runs one phase on its own.
"""

from rp1.utils.config import dispatch, run_hydra


def main() -> object:
    return run_hydra(dispatch, config_dir="training", config_name="posttrain")


if __name__ == "__main__":
    main()
