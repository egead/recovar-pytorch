import torch


def demean(x, axis=1):
    return x - torch.mean(x, dim=axis, keepdim=True)


def l2_normalize(x, eps=1e-27, axis=1):
    l2_norm = torch.sqrt(torch.mean(torch.square(x), dim=axis, keepdim=True))
    return x / (eps + l2_norm)


def l4_normalize(x, eps=1e-27, axis=1):
    l4_norm = torch.sqrt(torch.sqrt(torch.mean(torch.square(torch.square(x)), dim=axis, keepdim=True)))
    return x / (eps + l4_norm)


def l2_distance(x, y, axis=(1, 2)):
    x = demean(x)
    y = demean(y)
    return torch.sqrt(torch.mean(torch.square(x - y), dim=axis))


def l4_distance(x, y, axis=(1, 2)):
    x = demean(x)
    y = demean(y)
    return torch.sqrt(torch.sqrt(torch.mean(torch.square(torch.square(x - y)), dim=axis)))
