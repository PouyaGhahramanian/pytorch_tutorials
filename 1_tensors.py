
from __future__ import print_function
import torch
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

x = torch.zeros(2, 8, dtype = torch.double)
y = torch.randn_like(x, dtype = torch.double)

# Pytorch tensors and their corresponding numpy arrays share the same memory location
# Therefore, changing one will change the other
# x.to_numpy() && torch.from_numpy()
x_np = x.numpy()
logging.info('\tCorresponding numpy array is: {0}'.format(x_np))
# Equivalent to torch.add(x, y, out=x) ; x+y ; x = torch.add(..)
x.add_(y)
logging.info('\tCorresponding numpy array after addition is: {0}'.format(x_np))

# Numpy reshape ---> Pytorch view : y = x.view(-1, 8)
z = x.view(-1, 4)
logging.info('\tReshaped tensor size is: {0}'.format(z.size()))
print(z)
logging.info('\tReshaped tensor is: {0}'.format(z))

# Using Cuda device or CPU
if(torch.cuda.is_available()):
    logging.info('\tCuda device is available...')
    cuda_device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    p = torch.ones_like(x, device = cuda_device)
    x = x.to(cuda_device)
    xx = x.to(cpu_device, torch.float)
    logging.info('\tTensors with same valeu on cpu and cuda devices: ')
    print(p)
    print(x)
    print(xx)
else:
    logging.warn('\tCuda device not found.')
