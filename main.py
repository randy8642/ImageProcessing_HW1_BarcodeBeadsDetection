import argparse
import functions
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def main(srcPath, desPath):
    img = cv2.imread(srcPath)
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    hist, bin_edges = np.histogram(grayImg.flatten(), bins=range(256), density=True)
    newScale = 255 * np.cumsum(hist)
    newScale = np.concatenate([newScale, [newScale[-1]]], axis=0)
    newScale = newScale.astype(np.uint8)

    img = newScale[grayImg]
    print('avg filter')
    img = (functions.conv(img, np.ones([11, 11]))/121).astype(np.uint8)
    print('adpt. threshold')
   
    img = functions.adaptiveThreshold(img, kernalSize = 111)
    
    img = functions.erosion(img, np.ones([5, 5]))
    img = functions.erosion(img, np.ones([3, 3]))
   
    img = functions.dilation(img, np.ones([5, 5]))
    showImg(img)

    print('CCL')
    labels = functions.connectedComponents(img)
    
    uni, cnt = np.unique(labels, return_counts=True)

    tmp = np.zeros_like(img, dtype=np.uint8)
    for u, c in zip(uni[1:], cnt[1:]):
        if c > 200:
            tmp[labels == u] = 255
    img = tmp
    showImg(img)

    img = functions.erosion(img, np.ones([5, 5]))
    img = functions.dilation(img, np.ones([5, 5]))

    img[img != 0] = 255
    showImg(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(desPath, img)
    


def showImg(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--src', required=True)
    # parser.add_argument('--des', required=False)
    # args = parser.parse_args()

    # src = args.src
    # des = args.des
    
    # if des is None:
    #     des = os.path.join('./resultImages', os.path.basename(src)) 


    # main(src, des)
    main('./sourceImages/W_A1_0_3.jpg', './resultImages/W_A1_0_3.jpg')