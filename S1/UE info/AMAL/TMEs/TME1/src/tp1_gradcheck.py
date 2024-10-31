import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64) # q*p
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))

# q = 10
# p = 5

# n = 3 par exemple

#  TODO:  Test du gradient de Linear
x = torch.randn(10,3, requires_grad=True, dtype=torch.float64) # q*n
w = torch.randn(3,5, requires_grad=True, dtype=torch.float64) # n*p
b = torch.ones(5, requires_grad=True, dtype=torch.float64) # 1*p
torch.autograd.gradcheck(linear, (x, w, b))
