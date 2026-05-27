from mpmath.libmp.libelefun import exp_expneg_basecase
import torch
import math

from torch.nn import attention
def linear(x, W, b):
    return x @ W + b

x = torch.randn(4, 8)
W = torch.randn(8, 16)
b = torch.zeros(16)

Wk = torch.randn(8, 16)
Wv = torch.randn(8, 16)
bk = torch.zeros(16)
bv = torch.zeros(16)

q = linear(x, W, b)
k = linear(x, Wk, bk)
v = linear(x, Wv, bv)

scores = q @ k.transpose(-2, -1)

d_head = 16
scores = scores / math.sqrt(d_head)

x = scores
max_x = x.max(dim=-1, keepdim=True).values
x_shifted = x - max_x
exp_x = torch.exp(x_shifted)

sum_exp = exp_x.sum(dim=-1, keepdim=True)
attn = exp_x / sum_exp

out = attn @ v

print(out.shape)
print(attn.sum(dim=-1))
print(attn.shape)
print(max_x.shape)
print(exp_x.shape)
print(scores.shape)
print(k.shape) 
print(v.shape) 