"""Evaluate a policy on a benchmark: ``pixi run evaluate benchmark=<name>``."""

from rp1.utils.config import dispatch, run_hydra


def main() -> object:
    return run_hydra(dispatch, config_dir="inference", config_name="evaluate")


if __name__ == "__main__":
    main()
