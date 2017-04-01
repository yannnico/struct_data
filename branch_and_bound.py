# USAGE
# python sliding_window.py --image images/adrian_florida.jpg

# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
import numpy as np

from kernel import Kernel_A as Ker


image = cv2.imread('data/cows-images/cow-pic91-sml-lt.png')
(winW, winH) = (32, 32)

sh = np.shape(image)
Y = [list(range(sh[0])), list(range(sh[1])), list(range(sh[0])), list(range(sh[1]))]

P = []
V = []
Pm = Y
max = 0

kr = Ker(10, None)

print(sh)

while len(Pm[0]) > 1 or len(Pm[1]) > 1 or len(Pm[2]) > 1 or len(Pm[3]) > 1:

    lm = np.argmax(np.array([len(Pm[0]), len(Pm[1]), len(Pm[2]), len(Pm[3])]))

    mid = int(len(Pm[lm]) / 2)

    P0 = list(Pm)
    P1 = list(Pm)

    P0[lm] = Pm[lm][:mid]
    P1[lm] = Pm[lm][mid:]

    h0 = kr.k_x_ensemble(image, P0)
    h1 = kr.k_x_ensemble(image, P1)

    P.append(P0)
    P.append(P1)
    V.append(h0)
    V.append(h1)

    n_m = np.argmax(np.array(V))

    Pm = P.pop(n_m)
    V.pop(n_m)


print(Pm)
# since we do not have a classifier, we'll just draw the window
clone = image.copy()
cv2.rectangle(clone, (Pm[0][0], Pm[1][0]), (Pm[2][0], Pm[3][0]), (0, 255, 0), 2)
cv2.imshow("Window", clone)
cv2.waitKey()
