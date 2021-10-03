import torch
import numpy as np

# Tesnors from data
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# Tensors form numpy array
np_array = np.array(data)
t_from_np = torch.from_numpy(np_array)

# Tesnors from other tensors
# reatins the shape and dtype of x_data unless specified
x_ones = torch.ones_like(x_data)
print(x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

# shape is a tuple of tensor dimensions. In
# it determines the dimensionality of the output tensor.
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3,4)
print("Shape of tensor {}".format(tensor.shape))
print("Datatype of tensor {}".format(tensor.dtype))
print("Device tensor is stored on {}".format(tensor.device))

# We move our tensor to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Tensors can be sliced like numpy arrays
tensor = torch.ones(4,4)
print("First row", tensor[0,:])
print("First Column", tensor[:, 0])
print("Last column", tensor[:,-1])
tensor[:,1] = 0

# horizontal and vertical stcking of tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# matrix multiplication between two tensors y1, y2, y3
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# element wise multiplication
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

