import torch
from torchvision.transforms import v2

from rp1.training.harness.transforms import nested_resize


def test_nested_resize_matches_torchvision() -> None:
    image = torch.rand(3, 12, 10)
    sample = {"nested": {"image": image.clone(), "values": torch.tensor([-20.0, 2.0, 30.0])}}
    resized = nested_resize(6, "nested.image", "nested.image")(sample)
    assert torch.allclose(resized["nested"]["image"], v2.Resize(6)(image))
