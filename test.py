import torch


print(torch.cuda.device_count()) # 可用gpu数量
print(torch.cuda.is_available()) # 是否可用gpu
