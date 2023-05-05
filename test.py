#%%
import torch
import torch.nn as nn

a = torch.rand((1,3,4,4))
q = nn.AdaptiveMaxPool2d((1,1))
b = q(a)

print("a",a)
print("b",b)
#%%