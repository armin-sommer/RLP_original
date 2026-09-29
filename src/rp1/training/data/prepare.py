"""Prepare datasets and latent caches: ``pixi run prepare job=<name>``."""

from rp1.utils.config import dispatch, run_hydra


def main() -> object:
    return run_hydra(dispatch, config_dir="training/data", config_name="prepare")


if __name__ == "__main__":
    main()
