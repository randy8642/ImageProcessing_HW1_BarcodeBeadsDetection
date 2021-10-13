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

def conv(arr: np.ndarray):



    return 0

def erosion(img: np.ndarray, kernal: np.ndarray):
    assert len(img.shape) == 2, '輸入需為二值化圖像'
    assert (kernal.shape[0] % 2 == 1) & (kernal.shape[1] % 2 == 1), 'kernel需為奇數'
    
    # padding
    pad_x = kernal.shape[0] // 2
    pad_y = kernal.shape[1] // 2
    padImg = np.zeros([img.shape[0] + pad_x*2, img.shape[1] + pad_y*2], dtype=np.int8)
    padImg[pad_x:-pad_x, pad_y:-pad_y] = img
    
    newImg = np.zeros_like(img, dtype=np.int32)
    # conv
    for x, y in itertools.product(range(padImg.shape[0] - kernal.shape[0] + 1), range(padImg.shape[1] - kernal.shape[1] + 1)):             
        newImg[x, y] = np.sum(padImg[x:x+kernal.shape[0], y:y+kernal.shape[1]] * kernal)
    
    result = np.zeros_like(img, dtype=np.int8)
    result[newImg >= 9] = 1
   
    return result


def dilation(img: np.ndarray, kernal: np.ndarray):
    assert len(img.shape) == 2, '輸入需為二值化圖像'
    assert (kernal.shape[0] % 2 == 1) & (kernal.shape[1] % 2 == 1), 'kernel需為奇數'

    # padding
    pad_x = kernal.shape[0] // 2
    pad_y = kernal.shape[1] // 2
    newImg = np.zeros([img.shape[0] + pad_x*2, img.shape[1] + pad_y*2], dtype=np.int8)
    newImg[pad_x:-pad_x, pad_y:-pad_y] = img
    
    # conv
    for x, y in itertools.product(range(img.shape[0]), range(img.shape[1])):  
        if img[x, y]:            
            newImg[x:x+pad_x*2+1, y:y+pad_y*2+1] = np.logical_or(kernal, newImg[x:x+pad_x*2+1, y:y+pad_y*2+1])

    return newImg[pad_x:-pad_x, pad_y:-pad_y]



# img = np.ones([10, 10], dtype=np.int8)
# img[5, 5] = 0

# kernel = [[1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]]

# kernel = np.array(kernel)
# print(img)
# a = erosion(img, kernel)
# print(a)