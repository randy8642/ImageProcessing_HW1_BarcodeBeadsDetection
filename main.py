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
    img = functions.adaptiveThreshold(img, kernalSize = 111, offset=-3)

    # run erosion (7*7 kernel used)
    img = functions.erosion(img, np.ones([7, 7]))

    # run dilation 3 times (7*7 kernel used)
    for _ in range(3):
        img = functions.dilation(img, np.ones([7, 7]))
    
    # CCL
    labels = functions.connectedComponents(img)
    uni, cnt = np.unique(labels, return_counts=True)
    tmp = np.zeros(img.shape, dtype=np.uint8)
    tmp[labels > 0] = 1
    tmp[labels == uni[np.argmax(cnt[1:]) + 1]] = 0
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