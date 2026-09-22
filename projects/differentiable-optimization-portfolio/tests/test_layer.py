import numpy as np
import torch

from diffopt_portfolio.layer import MeanVarianceLayer


def test_layer_is_feasible_and_differentiable():
    cov = np.eye(3)
    layer = MeanVarianceLayer(cov)
    mu = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    w = layer(mu)
    assert torch.all(w >= -1e-5)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-4)
    loss = -(w * mu).sum()
    loss.backward()
    assert mu.grad is not None
    assert torch.isfinite(mu.grad).all()
