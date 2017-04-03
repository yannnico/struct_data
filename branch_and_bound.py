# USAGE
# python sliding_window.py --image images/adrian_florida.jpg

# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
import numpy as np

from kernel import Kernel_B as Ker
from kernel import ObjectifSVR as obj

SZ = 20
bin_n = 16  # Number of bins

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


image = cv2.imread('data/cows-images/cow-pic91-sml-lt.png')
(winW, winH) = (32, 32)

sh = np.shape(image)
Y = [list(range(sh[0])), list(range(sh[1])), list(range(sh[0])), list(range(sh[1]))]

P = []
V = []
Pm = Y
max = 0

with open('data/cows-labels/cow-pic91-sml-lt.txt') as input_file:
    for line in input_file:
        if 'Bounding' in line:
            item = line[70:-2]
            values = []
            for elem in item.split('-'):
                elem = elem.strip()
                elem = elem.strip(')')
                elem = elem.strip('(')
                print(elem.split(','))
                values.extend(elem.split(','))

x_min = int(values[0])
y_min = int(values[1])
x_max = int(values[2])
y_max = int(values[3])


kr = obj()

print(sh)

while len(Pm[0]) > 1 or len(Pm[1]) > 1 or len(Pm[2]) > 1 or len(Pm[3]) > 1:

    lm = np.argmax(np.array([len(Pm[0]), len(Pm[1]), len(Pm[2]), len(Pm[3])]))

    mid = int(len(Pm[lm]) / 2)

    P0 = list(Pm)
    P1 = list(Pm)

    P0[lm] = Pm[lm][:mid]
    P1[lm] = Pm[lm][mid:]

    f0_max = hog(image[P0[0][0]:P0[2][-1], P0[1][0]:P0[3][-1]])
    f1_max = hog(image[P1[0][0]:P1[2][-1], P1[1][0]:P1[3][-1]])

    if P0[0][-1] < P0[2][0] and P0[1][0] < P0[3][-1]:
        f0_min = hog(image[P0[0][-1]:P0[2][0], P0[1][-1]:P0[3][0]])
        h0 = kr.f_plus(f0_max) + kr.f_moins(f0_min)
    else:
        h0 = kr.f_plus(f0_max)

    if P0[0][-1] < P0[2][0] and P0[1][0] < P0[3][-1]:
        f1_min = hog(image[P1[0][-1]:P1[2][0], P1[1][-1]:P1[3][0]])
        h1 = kr.f_plus(f1_max) + kr.f_moins(f1_min)
    else:
        h1 = kr.f_plus(f1_max)

    P.append(P0)
    P.append(P1)
    V.append(h0)
    V.append(h1)

    n_m = np.argmax(np.array(V))

    Pm = P.pop(n_m)
    val = V.pop(n_m)

    print(len(Pm[0]))
    print(val)

    x_m_m = Pm[0][0]
    x_M_M = Pm[2][-1]
    y_m_m = Pm[1][0]
    y_M_M = Pm[3][-1]
    clone = image.copy()
    cv2.rectangle(clone, (Pm[0][0], Pm[1][0]), (Pm[2][-1], Pm[3][-1]), (0, 255, 0), 2)
    cv2.rectangle(clone, (Pm[0][-1], Pm[1][-1]), (Pm[2][0], Pm[3][0]), (0, 0, 255), 2)
    cv2.rectangle(clone, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time.sleep(0.3)

print(Pm)
# since we do not have a classifier, we'll just draw the window
clone = image.copy()
cv2.rectangle(clone, (Pm[0][0], Pm[1][0]), (Pm[2][0], Pm[3][0]), (0, 255, 0), 2)
cv2.rectangle(clone, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
cv2.imshow("Window", clone)
cv2.waitKey()
