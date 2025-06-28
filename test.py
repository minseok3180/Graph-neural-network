import torch
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
b = a.to('cuda:0')          # GPU로 이동
print(b[:, 1])              # ✅ OK