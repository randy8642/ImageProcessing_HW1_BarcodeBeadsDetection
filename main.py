import argparse
import functions
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def main(srcPath, desPath):
    img = cv2.imread(srcPath)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = functions.adaptiveThreshold(img, kernalSize = 111, offset=-3)
    img = functions.erosion(img, np.ones([7, 7]))
    for _ in range(3):
        img = functions.dilation(img, np.ones([7, 7]))
    

    labels = functions.connectedComponents(img)
    uni, cnt = np.unique(labels, return_counts=True)
    tmp = np.zeros(img.shape, dtype=np.uint8)
    tmp[labels > 0] = 1
    tmp[labels == uni[np.argmax(cnt[1:]) + 1]] = 0
    img = tmp

    img = np.logical_not(img).astype(np.uint8)*255
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