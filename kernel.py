import numpy as np


class Kernel_A:
    def __init__(self, dim, weight):
        self.weight = weight
        self.dim = dim
        self.ret = 0

    def k_x_y(self, image, y, t, l, b, r):
        res = np.zeros(self.dim)

        if y == 0:
            return 0

        return 0

    def k_x_ensemble(self, image, Y):
        t, l, b, r = Y[0], Y[1], Y[2], Y[3]

        x_0 = min(t)
        x_1 = max(b)
        y_0 = min(l)
        y_1 = max(r)

        h = max(0, x_1 - x_0)
        w = max(0, y_1 - y_0)

        if h * w == 0:
            return 0

        res = np.zeros(self.dim)
        p = np.random.rand()
        if p > 0.1:
            self.ret += 1
            return self.ret
        else:
            return self.ret-2
