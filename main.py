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
    img = functions.adaptiveThreshold(img, kernalSize = 51, offset=-3)

    # run erosion (5*5 kernel used)
    img = functions.erosion(img, np.ones([5, 5]))

    # run dilation (5*5 kernel used)
    img = functions.dilation(img, np.ones([5, 5]))
    
    # CCL
    labels = functions.connectedComponents(img)
    uni, cnt = np.unique(labels, return_counts=True)
    tmp = np.zeros(img.shape, dtype=np.uint8)
    for u, c in zip(uni[1:], cnt[1:]):
        if c > 40:
            tmp[labels == u] = 1
    img = tmp

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