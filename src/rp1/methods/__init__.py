"""Learned planning methods, one self-contained package each.

A method ``<name>`` is ``rp1.methods.<name>`` (its network, trainer and solver)
plus ``configs/methods/<name>/``: ``net.yaml``, ``train.yaml`` and
``solver.yaml``. ``pixi run posttrain training.method=<name>`` trains it on
the shared cache and value stages; ``core/agent/solver=<name>`` evaluates it.
"""
