import torch
import torch.nn.functional as F

from conv import Conv

def get_torch_conv(image, kernel, padding=0):
    return F.conv2d(image, kernel, padding=padding)

image = torch.tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]], dtype=torch.float)
image = torch.unsqueeze(image, 0)
image = torch.unsqueeze(image, 0)
kernel = torch.tensor([[1, 0, 1],[0, 1, 0],[1, 0, 1]], dtype=torch.float)
conv = Conv(3, kernel_tensor=kernel)
print(conv(image))

# check with PyTorch implementation
kernel = torch.unsqueeze(kernel, 0)
kernel = torch.unsqueeze(kernel, 0)
print(get_torch_conv(image, kernel, 1))