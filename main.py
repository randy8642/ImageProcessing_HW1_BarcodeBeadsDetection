import argparse
import functions
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def main():
    img = functions.readImages('./sourceImages')[0]
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    hist, bin_edges = np.histogram(grayImg.flatten(), bins=range(256), density=True)
    newScale = 255 * np.cumsum(hist)
    newScale = np.concatenate([newScale, [newScale[-1]]], axis=0)
    newScale = newScale.astype(np.uint8)

    img = newScale[grayImg]
    
    img = functions.conv(img, np.ones([11, 11])/121).astype(np.uint8)
    
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 5)

    img = functions.dilation(img, np.ones([3, 3]))

    img = functions.erosion(img, np.ones([5, 5]))

    # num_labels, labels = cv2.connectedComponents(img, connectivity=4)
    labels = functions.connectedComponents(img)
    
    uni, cnt = np.unique(labels, return_counts=True)

    tmp = np.zeros_like(img, dtype=np.uint8)
    for u, c in zip(uni[1:], cnt[1:]):
        if c > 300:
            tmp[labels == u] = 255
    img = tmp

    img = functions.dilation(img, np.ones([5, 5]))
    showImg(img)


def showImg(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__=='__main__':
    main()