import cv2
from matplotlib.pyplot import axis
import numpy as np
import os
import itertools

def readImages(folderPath:str) -> list:
    '''
    讀取檔案
    ---
    輸入資料夾名稱及可讀取內部所有JPG圖片\n
    並以回傳為列表
    '''

    fileList = sorted([f for f in os.listdir(folderPath) if f.endswith('jpg')])
    return [cv2.imread(os.path.join(folderPath, file)) for file in fileList]

def erosion(img: np.ndarray, kernal: np.ndarray):
    m, n = kernal.shape

    res = convWithPadding(img, kernal, 0).astype(np.uint8)

    res[res < m*n] = 0
    res[res >= m*n] = 1

    return res


def dilation(img: np.ndarray, kernal: np.ndarray):
    m, n = kernal.shape

    # create inverse mask of input
    img_inv = np.logical_not(img)

    # run erosion over inverse mask
    result_erosion = convWithPadding(img_inv, kernal, 1).astype(np.uint8)
    result_erosion[result_erosion < m*n] = 0
    result_erosion[result_erosion >= m*n] = 1

    # inverse mask again
    result = np.logical_not(result_erosion)

    return result


def convWithPadding(x: np.ndarray, y: np.ndarray, pad_value = 0) -> np.ndarray:

    # pad
    pad = np.array(y.shape) // 2
    padded_x = np.ones([x.shape[0] + pad[0]*2, x.shape[1] + pad[1]*2]) * pad_value
    padded_x[pad[0]:-pad[0], pad[1]:-pad[1]] = x
    
    # conv windows
    view_shape = tuple(np.subtract(padded_x.shape, y.shape) + 1) + y.shape
    strides = padded_x.strides + padded_x.strides
    sub_matrices = np.lib.stride_tricks.as_strided(padded_x, view_shape, strides)
    
    # for-loop method
    # n = np.zeros(sub_matrices.shape[:2])
    # for i in range(sub_matrices.shape[0]):
    #     for j in range(sub_matrices.shape[1]):
    #         n[i, j] = np.sum(np.multiply(sub_matrices[i, j, :, :], y))

    # einsum method
    m = np.einsum('ij,klij->kl', y, sub_matrices)

    return m

def connectedComponents(img: np.ndarray):
    assert len(img.shape) == 2

    padImg = np.zeros([img.shape[0] + 1 * 2, img.shape[1] + 1 * 2], dtype=bool)
    padImg[1:-1, 1:-1] = img.astype(bool)

    # 
    connectedLabel = [0]
    mask = np.zeros_like(padImg, dtype=np.int32)
    for i in range(1, padImg.shape[0], 1):
        for j in range(1, padImg.shape[1], 1):
            if not padImg[i, j]:
                continue
            
            upper = mask[i-1, j]
            lefter = mask[i, j-1]

            if upper and (not lefter):
                mask[i, j] = upper
            elif (not upper) and lefter:
                mask[i, j] = lefter
            elif upper and lefter:
                mask[i, j] = upper
                connectedLabel[lefter] = upper
            elif (not upper) and (not lefter):
                mask[i, j] = len(connectedLabel)
                connectedLabel.append(len(connectedLabel))

    # 
    for n in range(1, len(connectedLabel)):
        c = n
        while c != connectedLabel[c]:
            c = connectedLabel[c]

        mask[mask == n] = c

    return mask[1:-1, 1:-1]

def adaptiveThreshold(x:np.ndarray, kernalSize=3, offset = -5):

    sigma = 0.3 * ((kernalSize - 1) * 0.5 - 1) + 0.8
    guass_kernal = get_guassKernal(l=kernalSize, sig=sigma)
    threshold = convWithPadding(x, guass_kernal, 0) + offset

    res = np.zeros(x.shape, dtype=np.uint8)
    res[x < threshold] = 1

    return res

def get_guassKernal(l=5, sig=1.) -> np.ndarray:
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

