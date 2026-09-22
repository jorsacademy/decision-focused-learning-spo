from pathlib import Path

import torch

from diffopt_portfolio.layer import MeanVarianceLayer
from diffopt_portfolio.model import EndToEndPortfolio, realized_mean_variance_loss
from diffopt_portfolio.synthetic import generate_contextual_returns

x, y, covariance = generate_contextual_returns()
x_t, y_t = torch.tensor(x), torch.tensor(y)
layer = MeanVarianceLayer(covariance, risk_aversion=1.0)
model = EndToEndPortfolio(x.shape[1], y.shape[1], layer)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(60):
    opt.zero_grad()
    w = model(x_t)
    loss = realized_mean_variance_loss(w, y_t, risk_penalty=0.01)
    loss.backward()
    opt.step()

Path("checkpoints").mkdir(exist_ok=True)
torch.save(model.forecaster.state_dict(), "checkpoints/forecaster.pt")
print(f"loss={loss.item():.6f}")
