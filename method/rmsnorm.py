import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 γ
        # print(self.weight.shape)# torch.Size([4])

    def _norm(self, x):
        # 均方根归一化：沿最后一维计算
        # torch.rsqrt返回的是x.pow(2).mean(-1, keepdim=True) + self.eps的平方根的倒数
        # 直接调用 rsqrt 比先 sqrt 再 1 / 更高效，尤其在 GPU 上
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        # print(self._norm(x.float()).shape)# # torch.Size([1, 2, 4])
        return self.weight * self._norm(x.float()).type_as(x)


# 实例化测试
x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0]]])  # shape = (1, 2, 4)
norm = RMSNorm(dim=4)
output = norm(x)
print(output.shape)# torch.Size([1, 2, 4])