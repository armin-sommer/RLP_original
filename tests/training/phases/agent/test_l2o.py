from rp1.training.phases.agent.l2o import dagger_beta


def test_dagger_beta_follows_the_paper_schedule() -> None:
    steps, rounds, decay = 2000, 20, 0.8
    assert dagger_beta(0, steps, rounds, decay) == 1.0  # round 0: pure expert
    assert abs(dagger_beta(100, steps, rounds, decay) - 0.8) < 1e-9
    assert abs(dagger_beta(1999, steps, rounds, decay) - 0.8**19) < 1e-9
    assert dagger_beta(150, steps, rounds, decay) == dagger_beta(199, steps, rounds, decay)
