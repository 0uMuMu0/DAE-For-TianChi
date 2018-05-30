from collections import defaultdict

import csv
import sys

import cv2

from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

import matplotlib.pyplot as plt
from matplotlib import cm

FILE_2015 = 'preliminary/quickbird2015.tif'
FILE_2017 = 'preliminary/quickbird2017.tif'
FILE_cadastral2015 = '20170907_hint/cadastral2015.tif'
FILE_tinysample = '20170907_hint/tinysample.tif'

im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
im_tiny = tiff.imread(FILE_tinysample)
im_cada = tiff.imread(FILE_cadastral2015)

# im_2015.shape  (5106, 15106, 4)
# im_2017.shape  (5106, 15106, 4)
# im_tiny.shape  (5106, 15106, 3)
# im_cada.shape  (5106, 15106)


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w*h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0)
    lens = maxs - mins
    matrix = (matrix - mins[None, :]) / lens[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

p1 = plt.subplot(121)
i1 = p1.imshow(scale_percentile(im_2015[100:1000, 100:1000, :3]))
# plt.colorbar(i1)

p2 = plt.subplot(122)
i2 = p2.imshow(im_2015[100:1000, 100:1000, 3])
# plt.colorbar(i2)

# plt.show()

x_start = 500
x_end = 1000
y_start = 4800
y_end = 5300

plt.subplots(ncols=3, nrows=2, figsize=(16, 8))

mt = np.ma.array(np.ones((x_end-x_start, y_end-y_start)),
                 mask=((im_tiny[x_start:x_end, y_start:y_end, 0]/np.max(im_tiny)+im_cada[x_start:x_end, y_start:y_end])==0))

p151 = plt.subplot(231)
i151 = p151.imshow(im_2015[x_start:x_end, y_start:y_end, 3])
plt.colorbar(i151)

p152 = plt.subplot(233)
i152 = p152.imshow(im_tiny[x_start:x_end, y_start:y_end, 0])
plt.colorbar(i152)

p153 = plt.subplot(232)
i153 = p153.imshow(im_2015[x_start:x_end, y_start:y_end, 3])
p153.imshow(mt, cmap=cm.bwr, alpha=0.3, vmin=0, vmax=1)
plt.colorbar(i153)

p171 = plt.subplot(234)
i171 = p171.imshow(im_2017[x_start:x_end, y_start:y_end, 3])
plt.colorbar(i171)

p172 = plt.subplot(236)
i172 = p172.imshow(im_cada[x_start:x_end, y_start:y_end])
plt.colorbar(i172)

p173 = plt.subplot(235)
i173 = p173.imshow(im_2017[x_start:x_end, y_start:y_end, 3])
p173.imshow(mt, cmap=cm.bwr, alpha=0.3, vmin=0, vmax=1)
plt.colorbar(i173)

plt.show()
plt.show()
"""
plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

p1 = plt.subplot(241)
i1 = p1.imshow(im_2015[500:1000, 10200:10700, 0])
plt.colorbar(i1)

p2 = plt.subplot(242)
i2 = p2.imshow(im_2015[500:1000, 10200:10700, 1])
plt.colorbar(i2)

p3 = plt.subplot(243)
i3 = p3.imshow(im_2015[500:1000, 10200:10700, 2])
plt.colorbar(i3)

p4 = plt.subplot(244)
i4 = p4.imshow(im_2015[500:1000, 10200:10700, 3])
plt.colorbar(i4)

p5 = plt.subplot(245)
i5 = p5.imshow(im_2017[500:1000, 10200:10700, 0])
plt.colorbar(i5)

p6 = plt.subplot(246)
i6 = p6.imshow(im_2017[500:1000, 10200:10700, 1])
plt.colorbar(i6)

p7 = plt.subplot(247)
i7 = p7.imshow(im_2017[500:1000, 10200:10700, 2])
plt.colorbar(i7)

p8 = plt.subplot(248)
i8 = p8.imshow(im_2017[500:1000, 10200:10700, 3])
plt.colorbar(i8)

plt.show()

"""







