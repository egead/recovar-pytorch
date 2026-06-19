import torch


def diff(x, axis):
    n = x.shape[axis]
    a = torch.narrow(x, axis, 1, n - 1)
    b = torch.narrow(x, axis, 0, n - 1)
    return a - b


def cubic_interp1d(x0, x, y):
    size = x.shape[0]

    xdiff = diff(x, axis=0)
    ydiff = diff(y, axis=0)

    Li = [None] * size
    Li_1 = [None] * (size - 1)
    z = [None] * size

    Li[0] = torch.sqrt(2.0 * xdiff[0])
    Li_1[0] = torch.zeros_like(xdiff[0])
    B0 = torch.zeros_like(xdiff[0])
    z[0] = B0 / Li[0]

    for i in range(1, size - 1, 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = torch.sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6.0 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = torch.sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bi = torch.zeros_like(xdiff[0])
    z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    z = torch.stack(z)

    index = torch.searchsorted(x, x0)
    index = torch.clip(index, 1, size - 1)

    xi1 = torch.gather(x, 0, index)
    xi0 = torch.gather(x, 0, index - 1)
    yi1 = torch.gather(y, 0, index)
    yi0 = torch.gather(y, 0, index - 1)
    zi1 = torch.gather(z, 0, index)
    zi0 = torch.gather(z, 0, index - 1)
    hi1 = xi1 - xi0

    f0 = (
        zi0 / (6 * hi1) * (xi1 - x0) ** 3
        + zi1 / (6 * hi1) * (x0 - xi0) ** 3
        + (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0)
        + (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
    )
    return f0
