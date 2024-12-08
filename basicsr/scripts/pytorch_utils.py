import torch


def channel_randperm(x):
    """
    对 tensor 的通道进行随机排列。

    Args:
        x (torch.Tensor): 输入 tensor，形状为 [N, C, H, W]

    Returns:
        torch.Tensor: 通道随机排列后的 tensor
    """
    N, C, H, W = x.shape
    perm = torch.randperm(C)
    x = x[:, perm, :, :]
    return x
