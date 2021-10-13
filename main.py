import argparse
import functions
import cv2
import numpy as np

def main():
    img = functions.readImages('./sourceImages')[0]
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('gray', grayImg)
    # cv2.waitKey(0)

    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1],]

    eImg = functions.erosion(grayImg, np.array(kernel))
    closingImg = functions.dilation(eImg, np.array(kernel))

    cv2.imshow('close', cv2.resize(closingImg.astype(np.uint8), (800, int(800/closingImg.shape[1]*closingImg.shape[0]))))
    cv2.waitKey(0)

    pass


if __name__=='__main__':
    main()