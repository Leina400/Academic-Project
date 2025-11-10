import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))
print("Gradcheck MSE :", torch.autograd.gradcheck(mse, (yhat, y)))

#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)

batch, in_dim, out_dim = 4, 3, 2
X = torch.randn(batch, in_dim, dtype=torch.float64, requires_grad=True)
W = torch.randn(in_dim, out_dim, dtype=torch.float64, requires_grad=True)
b = torch.randn(1, out_dim, dtype=torch.float64, requires_grad=True)

print("Gradcheck Linear :", torch.autograd.gradcheck(linear, (X, W, b)))

