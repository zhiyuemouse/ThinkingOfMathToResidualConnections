import numpy as np

# 用一个极简的一维例子说明：残差连接如何给梯度多开一条“直通路”
# 普通层：y = w * x
# 残差层：y = w * x + x = (w + 1) * x

w_list = [0.5, 0.1, 0.01, 0.0]
num_layers = 10

print("假设网络有 10 层，每一层的权重都是 w")
print("普通网络的整体梯度：w^10")
print("残差网络的整体梯度：(w + 1)^10")
print("-" * 60)

for w in w_list:
    grad_plain = w ** num_layers
    grad_residual = (w + 1) ** num_layers
    print(f"w = {w}")
    print(f"  普通网络梯度     = {grad_plain:.12f}")
    print(f"  残差连接后的梯度 = {grad_residual:.12f}")
    print()

print("可以看到：当 w 很小时，普通网络的梯度会因为连乘迅速接近 0。")
print("但加入残差连接后，每层导数从 w 变成 w + 1，梯度就多了一条更容易传回去的路。")
