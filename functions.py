import cv2
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

    res = conv(img, kernal, 0).astype(np.uint8)

    res[res < m*n] = 0
    res[res >= m*n] = 1

    return res


def dilation(img: np.ndarray, kernal: np.ndarray):
    m, n = kernal.shape

    img_inv = np.logical_not(img)

    res = conv(img_inv, kernal, 1).astype(np.uint8)
    res[res < m*n] = 1
    res[res >= m*n] = 0

    return res


def conv(x: np.ndarray, y: np.ndarray, pad_value = 0) -> np.ndarray:

    # pad
    pad = np.array(y.shape) // 2
    padded_x = np.ones([x.shape[0] + pad[0]*2, x.shape[1] + pad[1]*2]) * pad_value
    padded_x[pad[0]:-pad[0], pad[1]:-pad[1]] = x
    

    # conv windows
    # divide the matrix into sub_matrices of kernel size
    view_shape = tuple(np.subtract(padded_x.shape, y.shape) + 1) + y.shape
    strides = padded_x.strides + padded_x.strides
    sub_matrices = np.lib.stride_tricks.as_strided(padded_x, view_shape, strides)
    # convert non_zero elements to 1 (dummy representation)
    # sub_matrices[sub_matrices > 0.] = 1.
    
    m = np.einsum('ij,klij->kl', y, sub_matrices)

    return m

def connectedComponents(x:np.ndarray):
    labels = [0]
    connected = [0]
    flag = np.zeros_like(x, dtype=np.int32)
    for n, (i, j) in enumerate(itertools.product(range(x.shape[0]), range(x.shape[1]))):
        if x[i, j] == 0:
            continue
       
        if (i > 0) and (j > 0) and (flag[i-1, j] != 0) and (flag[i, j-1] != 0):
            connected[flag[i, j-1]] = flag[i-1, j]
            flag[i, j] = flag[i-1, j]
        elif ((flag[i-1, j] != 0) and (i > 0)) and ((flag[i, j-1] == 0) or (j == 0)):
            flag[i, j] = flag[i-1, j]
        elif ((flag[i-1, j] == 0) or (i == 0)) and ((flag[i, j-1] != 0) and (j > 0)):
            flag[i, j] = flag[i, j-1]
        else:
            labels.append(labels[-1] + 1)
            connected.append(labels[-1])
            flag[i, j] = labels[-1]

    for l in range(1, len(labels)):
        c = l
        while c != connected[c]:
            c = connected[c]
        flag[flag == l] = c

    return flag

def adaptiveThreshold(x:np.ndarray, kernalSize=3, offset = -5):

    sigma = 0.3 * ((kernalSize - 1) * 0.5 - 1) + 0.8
    guass_kernal = get_guassKernal(l=kernalSize, sig=sigma)
    threshold = conv(x, guass_kernal, 0) + offset

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


