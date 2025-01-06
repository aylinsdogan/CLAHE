import math
import numpy as np
import cv2 as cv
import random
from matplotlib import pyplot as plt

lut = [] # look-up-table arrayi
# INTERPOLATION
def interpolation(img,x,y):
    if len(img.shape) == 3:
        rows, cols, dim = img.shape
        r = int(cols / y)
        n0 = 0
        n1 = r
        output = np.zeros((rows, cols, dim), dtype=np.uint8)
        for i in range(int(x/2), rows-int(x/2), x):
            for j in range(int(y/2), cols-int(y/2), y):
                if j==int(y/2):
                    n=n1
                sub = img[i:i + x, j:j + y,2]
                output[i:i + x, j:j + y,2] = interpolationSub(sub, lut[n0], lut[n0 + 1], lut[n1],lut[n1 + 1], x, y)
                output[i:i + x, j:j + y, 0] = img[i:i + x, j:j + y, 0]
                output[i:i + x, j:j + y, 1] = img[i:i + x, j:j + y, 1]
                n0 += 1
                n1 += 1
            n0=n
            n1=n+r
    else:
        rows, cols = img.shape
        r = int(cols / x)
        n0 = 0
        n1 = r
        output = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(int(x / 2), rows - x, x):
            for j in range(int(y / 2), cols - y, y):
                if j == int(y / 2):
                    n = n1
                sub = img[i:i + x, j:j + y]
                output[i:i + x, j:j + y] = interpolationSub(sub, lut[n0], lut[n0 + 1], lut[n1], lut[n1 + 1], x,y)
                n0 += 1
                n1 += 1
            n0 = n
            n1 = n + r
    return output
# CALCULATION OF INTERPOLATION
def interpolationSub(sub,LU,RU,LB,RB,x,y):
    subImage = np.zeros(sub.shape)
    num = x*y
    for i in range(x):
        inverseI = x - i
        for j in range(y):
            inverseJ = y - j
            val = sub[i, j].astype(int)
            subImage[i, j] = int(np.round((inverseI * (inverseJ * LU[val] + j * RU[val]) + i * (inverseJ * LB[val] + j * RB[val])) / float(num)))
    return subImage
# CLAHE
def clahe(img,cliplimit,x,y):
    if len(img.shape) < 3:
        # PADDING
        r, c = img.shape
        pdr1= math.floor(float(x-r%x)/2)
        pdr2= math.ceil(float(x-r%x)/2)
        pdc1= math.floor(float(y-c%y)/2)
        pdc2= math.ceil(float(y-c%y)/2)
        img = cv.copyMakeBorder(img, pdr1,pdr2, pdc1 , pdc2, cv.BORDER_REFLECT)
        img = cv.copyMakeBorder(img, x, x, y, y, cv.BORDER_REFLECT)
        rows, cols = img.shape
        output = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(0,rows,x):
            for j in range(0,cols,y):
                output[i:i+x,j:j+y] = histogram(img[i:i+x,j:j+y],cliplimit)
    else:
        # PADDING
        r, c, d = img.shape
        pdr1 = math.floor(float(x - r % x) / 2)
        pdr2 = math.ceil(float(x - r % x) / 2)
        pdc1 = math.floor(float(y - c % y) / 2)
        pdc2 = math.ceil(float(y - c % y) / 2)
        img = cv.copyMakeBorder(img, pdr1, pdr2, pdc1, pdc2, cv.BORDER_REFLECT)
        img = cv.copyMakeBorder(img, x, x, y, y, cv.BORDER_REFLECT)
        rows, cols, dim = img.shape
        output = np.zeros((rows, cols, dim), dtype=np.uint8)
        for i in range(0, rows, x):
            for j in range(0, cols, y):
                output[i:i + x, j:j + y,2] = histogram(img[i:i + x, j:j + y,2], cliplimit)
                output[i:i + x, j:j + y, 1] = img[i:i + x, j:j + y, 1]
                output[i:i + x, j:j + y, 0] = img[i:i + x, j:j + y, 0]
    output = interpolation(img, x, y)
    output = output[x:-x,y:-y]
    return output[pdr1:-pdr2,pdc1:-pdc2]
# HISTOGRAM
def histogram(img,cliplimit):
    maxClipSize=100
    rows, cols = img.shape
    pdf = [0]*256
    cdf = [0]*256
    result = [0]*256
    output = 255 * np.ones_like(img, dtype=np.uint8)
    size = cols*rows
    L = 256
    # calculating pdf
    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            pdf[k] += 1
    # CLIPPING HISTOGRAM
    cliplimit = int(cliplimit * max(pdf) / maxClipSize)
    if cliplimit > 0:
        diff = [0]
        diffIndex = [0] * 256
        for i in range(L):
            if pdf[i] > cliplimit:
                diff.append(int(pdf[i] - cliplimit))
                diffIndex[i] += 1
                pdf[i] = cliplimit
        diffsum = np.sum(diff)
        for i in range(diffsum):
            rand = random.randint(0,255)
            if diffIndex[rand] == 0 and pdf[rand] < cliplimit:
                pdf[rand] += 1
            else:
                i-=1
    # calculating cdf
    sum = 0
    for i in range(L):
        sum += pdf[i]
        cdf[i] = sum
    for i in range(L):
        result[i] = np.round((cdf[i]/size * (L - 1)))
    lut.append(result)
    for i in range(rows):
        for j in range(cols):
            output[i, j] = result[img[i, j]]
    return output
###################################################
# MY CLAHE
"""
img = cv.imread('1.jpg', cv.IMREAD_COLOR)
hsvimg = cv.cvtColor(img, cv.COLOR_RGB2HSV)
clahe = clahe(hsvimg,40,64,64)
clahe = cv.cvtColor(clahe, cv.COLOR_HSV2RGB)
cv.imshow("My clahe",clahe)

# OPENCV
img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
clahe = cv.createCLAHE(clipLimit=4,tileGridSize=(8,8))
img[:,:,0] = clahe.apply(img[:,:,0])
img = cv.cvtColor(img, cv.COLOR_LAB2RGB)
cv.imshow("openCV's clahe", img)
cv.waitKey(0)
"""
