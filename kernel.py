import numpy as np
import h5py

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


class Kernel_B:
    def __init__(self, dim, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.s = (y_max-y_min) * (x_max-x_min)
        self.dim = dim
        self.ret = 0

    def k_x_y(self, image, y, t, l, b, r):

        if y == 0:
            return 0

        s1 = max((b-t), 0)*max((r-l), 0)

        if s1 == 0:
            return 0
        s2 = self.s

        s3 = (max(min(self.x_max, b) - max(self.x_min, t), 0)) * (max(min(self.y_max, r) - max(self.y_min, l), 0))

        if s3 == 0:
            return 0

        if s2 + s1 - s3 == 0:
            print("here")
            print(s1)
            print(s2)
            print(s3)
            print(min(self.x_max, b))
            print(max(self.x_min, t))
            print(min(self.y_max, r))
            print(max(self.y_min, l))
            return 0

        if s3 > s1:
            print("booh")

        return s3 / s2

    def k_x_y_m(self, image, y, t, l, b, r):

        if y == 0:
            return 0

        s1 = max((b-t), 0)*max((r-l), 0)

        if s1 == 0:
            return 0
        s2 = self.s

        s3 = (max(min(self.x_max, b) - max(self.x_min, t), 0)) * (max(min(self.y_max, r) - max(self.y_min, l), 0))

        if s3 == 0:
            return 0

        if s2 + s1 - s3 == 0:
            print("here")
            print(s1)
            print(s2)
            print(s3)
            print(min(self.x_max, b))
            print(max(self.x_min, t))
            print(min(self.y_max, r))
            print(max(self.y_min, l))
            return 0

        if s3 > s1:
            print("booh")

        return (s1 - s3) / (s2 + s1 - s3)

    def k_x_ensemble(self, image, Y):
        t, l, b, r = Y[0], Y[1], Y[2], Y[3]

        x_m_m = t[0]
        x_m_M = t[-1]
        x_M_m = b[0]
        x_M_M = b[-1]
        y_m_m = l[0]
        y_m_M = l[-1]
        y_M_m = r[0]
        y_M_M = r[-1]

        return self.k_x_y(image, 1, x_m_m, y_m_m, x_M_M, y_M_M) - self.k_x_y_m(image, 1, x_m_M, y_m_M, x_M_m, y_M_m)


class ObjectifSVR:
    def __init__(self):
        h5f = h5py.File("coef.h5", 'r')
        self.coef = np.array(h5f['coef'])
        h5f.close()

    def f_(self, v):
        np.dot(v, self.coef)

    def f_plus(self, v):
        r = np.multiply(v, self.coef)
        return np.sum((np.abs(r) + r)) / 2

    def f_moins(self, v):
        r = np.multiply(v, self.coef)
        return np.sum((- np.abs(r) + r)) / 2
