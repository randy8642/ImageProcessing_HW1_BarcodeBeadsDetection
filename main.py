import argparse
import functions
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def main(srcPath, desPath):
    img = cv2.imread(srcPath)
    
    # convert to grayscale
    img = img[:, :, 2]*0.299 + img[:, :, 1]*0.587 + img[:, :, 0]*0.114
    img = img.astype(np.uint8)
    
    # transfer to binary image by adaptive threshold method
    img = functions.adaptiveThreshold(img, kernalSize = 51, offset=-4)

   
    # create mask 1
    tmp = functions.erosion(img, np.ones([3, 3]))
    tmp = functions.dilation(tmp, np.ones([5, 5]))
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)

    mask_1 = np.zeros_like(labels, dtype=np.uint8)
    for n, s in enumerate(stats):
        if n == 0:
            continue
        if s[-1] > 500:
            mask_1[labels == n] = 1
    mask_1 = functions.dilation(mask_1, np.ones([5, 5]))
    mask_1 = np.logical_not(mask_1)


    # create mask 2
    tmp = functions.erosion(img, np.ones([3, 3]))
    tmp = functions.dilation(tmp, np.ones([31, 31]))
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)

    mask_2 = np.zeros_like(labels, dtype=np.uint8)
    for n, s in enumerate(stats):
        if n == 0:
            continue
        if (s[2] > 200) or (s[3] > 200):
            mask_2[labels == n] = 1
    mask_2 = np.logical_not(mask_2)

    # masked
    mask = np.logical_and(mask_1, mask_2)
    
    img = np.logical_and(img, mask) * 1
    img = img.astype(np.uint8)


    # run erosion (5*5 kernel used)
    img = functions.erosion(img, np.ones([5, 5]))

    # run dilation (5*5 kernel used)
    img = functions.dilation(img, np.ones([5, 5]))
    

    # reverse black and white
    img = np.logical_not(img).astype(np.uint8) * 255

    # convert grayscale to RGB
    img = np.concatenate([np.expand_dims(img, axis=-1)] * 3, axis=-1)

    # save image to disk
    cv2.imwrite(desPath, img)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--des', required=False)
    args = parser.parse_args()

    src = args.src
    des = args.des
    
    if des is None:
        des = os.path.join('./resultImages', os.path.basename(src)) 


    main(src, des)