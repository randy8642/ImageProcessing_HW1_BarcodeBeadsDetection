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
    assert len(img.shape) == 2, '輸入需為二值化圖像'
    assert (kernal.shape[0] % 2 == 1) & (kernal.shape[1] % 2 == 1), 'kernel需為奇數'
    

    res = conv(img, kernal).astype(np.uint8)

    res[res < len(kernal.flatten())] = 0
    res[res >= len(kernal.flatten())] = 1
    

    return res


def dilation(img: np.ndarray, kernal: np.ndarray):
    m, n = kernal.shape

    assert len(img.shape) == 2, '輸入需為二值化圖像'
    assert (m % 2 == 1) & (n % 2 == 1), 'kernel需為奇數'
    
    f_img = img.reshape(-1, 1)
    b = f_img * kernal.flatten()   
    b = b.reshape(-1, m, n).astype(np.uint8)

    res = np.zeros([img.shape[0] + m//2 *2, img.shape[1] + n//2 *2], dtype=np.uint8)
    for n, (i, j) in enumerate(itertools.product(range(res.shape[0]-m+1), range(res.shape[1]-n+1 ))):     
        res[i:i+kernal.shape[0], j:j+kernal.shape[1]] = np.logical_or(b[n], res[i:i+kernal.shape[0], j:j+kernal.shape[1]])

    return res


def conv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # pad
    pad = np.array(y.shape) // 2
    padded_x = np.zeros([x.shape[0] + pad[0]*2, x.shape[1] + pad[1]*2])
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

def ad(x:np.ndarray):
    # https://cloud.tencent.com/developer/ask/72570

    # 高斯conv. avg.
    # threshold

    return x