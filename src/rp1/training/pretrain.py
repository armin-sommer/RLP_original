"""Pretrain a world model: ``pixi run pretrain``."""

from rp1.utils.config import dispatch, run_hydra


def main() -> object:
    return run_hydra(dispatch, config_dir="training", config_name="pretrain")


if __name__ == "__main__":
    main()
