import torch
import jittor as jt
import numpy as np

# 生成随机张量 (512, 512)
torch.manual_seed(42)
x_torch = torch.randn(512, 512)
y_torch = torch.randn(512, 512)

# PyTorch 计算整体 L2 距离
dist_torch = torch.dist(x_torch, y_torch, 2)

# 转换为 Jittor 张量
x_jt = jt.array(x_torch.numpy())  
y_jt = jt.array(y_torch.numpy())

# Jittor 计算整体 L2 距离
dist_jt = (x_jt - y_jt).pow(2).sum().sqrt()

# 打印结果对比
print(f"PyTorch dist: {dist_torch.item()}")
print(f"Jittor dist: {dist_jt.item()}")

# 计算误差
error = abs(dist_torch.item() - dist_jt.item())
print(f"Error: {error}")

# 断言误差在合理范围内
assert error < 1e-5, "Jittor and PyTorch results do not match!"
print("✅ Jittor implementation matches PyTorch!")
