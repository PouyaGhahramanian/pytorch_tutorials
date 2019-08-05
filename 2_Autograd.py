
from __future__ import print_function
import torch
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

x = torch.ones(2, 2, requires_grad = True)
y = x + 2
logging.info('\tGrad function for y is: {0}'.format(y.grad_fn))

z = y * y * 3
out = z.mean()
logging.info('\tZ and Out tensor values are: {0}, {1}'.format(z, out))

# Default value of requires_grad is False, unless the variable is created by an operation
a = torch.randn(2, 2, dtype = torch.float)
a = (a * 3) / (a - 1)
logging.info('\trequires_grad value of a is: {0}'.format(a.requires_grad))
a.requires_grad_(True)
logging.info('\trequires_grad value of a is: {0}'.format(a.requires_grad))
b = (a * a).sum()
logging.info('\trequires_grad value of b is: {0}'.format(b.requires_grad))
logging.info('\tgrad_fn value of b is: {0}'.format(b.grad_fn))

# Backpropaggation is too simple ;)
# Backpropaggate out with respect to x
out.backward()
logging.info('\td(out)/dx after backward from out is: {0}'.format(x.grad))

# torch.autograd is an engine for computing vector-Jacobian product -- Chain Rule --
x = torch.randn(3, requires_grad=True)
y = x * 2
while(y.data.norm() < 1000):
    y = y * y
logging.info('\ty: {0}'.format(y))

# Chain Rule
v = torch.tensor([0.1, 1, 0.001], dtype = torch.float)
y.backward(v)
logging.info('\tBackward with chain rule wrt x: {0}'.format(x.grad))

# Setting torch.require grad to False using torch.no_grad():
logging.info(x.requires_grad)
logging.info((x ** 2).requires_grad)
with torch.no_grad():
    logging.info((x ** 2).requires_grad)
