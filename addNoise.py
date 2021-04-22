import numpy as np
import cv2
import random
import glob

from PIL import Image


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''

    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]

    return output


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''

    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)

    return out


for jpgfile in glob.glob(r'D:\demo\PyPro\TJGradutionProData\img\*.jpg'):
    img = cv2.cvtColor(cv2.imread(jpgfile), cv2.COLOR_RGB2BGR)
    # 添加椒盐噪声，噪声比例为 0.02
    out1 = sp_noise(img, prob=0.02)
    # 添加高斯噪声，均值为0，方差为0.01
    out2 = gasuss_noise(out1, mean=0, var=0.01)
    img_name = jpgfile.split('\\')[-1]
    out2 = Image.fromarray(out2)
    out2.save(r'D:\demo\PyPro\TJGradutionProData\img-noise\{0}'.format(img_name))
