import torch

from rp1.data import LatentCache


def test_latent_cache_windowing_is_causal_and_episode_local() -> None:
    cache = LatentCache(
        z=torch.arange(10, dtype=torch.float32).view(5, 2),
        episode_idx=torch.tensor([0, 0, 0, 1, 1]),
        step_idx=torch.tensor([0, 1, 2, 0, 1]),
    )
    windowed = cache.windowed(frames=3, lag=1)
    assert windowed.z.shape == (5, 6)
    assert torch.equal(windowed.z[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(windowed.z[2], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(windowed.z[3], torch.tensor([6.0, 7.0, 6.0, 7.0, 6.0, 7.0]))


def test_first_episodes_caps_source_episodes_of_a_multiplexed_cache() -> None:
    cache = LatentCache(
        z=torch.randn(40, 2),
        episode_idx=torch.arange(40) // 4,
        step_idx=torch.arange(40) % 4,
        meta={"phase_multiplex": 2},
    )
    capped = cache.first_episodes(3)
    assert sorted(capped.episodes()) == [0, 1, 2, 3, 4, 5]
    assert capped.meta is not None and capped.meta["max_episodes"] == 3
    assert cache.first_episodes(None) is cache
