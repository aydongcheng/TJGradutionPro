import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


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


img = cv2.imread("1_0_0_20161219154724341.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 添加椒盐噪声，噪声比例为 0.02
out1 = sp_noise(img, prob=0.02)

# 添加高斯噪声，均值为0，方差为0.01
out2 = gasuss_noise(out1, mean=0, var=0.01)

# 显示图像
titles = ['Original Image', 'Add Salt and Pepper noise', 'Add Gaussian noise']
images = [img, out1, out2]

plt.figure(figsize=(20, 15))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


def filter2d(im, kernel):
    '''
    入口参数：数组格式图像im, 卷积核kernel
    经过2D卷积操作完成去噪平滑
    '''
    m, n = im.shape # 获取图片长宽
    result = np.zeros(im.shape) # 创建0元素矩阵
    w = kernel.shape[0]
    l = (w-1) // 2
    for x in range(l, m-l):
        for y in range(l, n-l):
            # im[a:b, c:d]行列切片操作
            result[x, y] = (im[x-l:x+l+1, y-l:y+l+1]*kernel).sum()
    return result

# 卷积核大小n须为奇数，这里采用均值滤波
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])*1/9

imout = filter2d(images[2], kernel)

plt.gray() # 灰度格式输出
plt.subplot(1,2,1) # 1x2的图幅中的第1张图
plt.imshow(images[2])
plt.subplot(1,2,2) # 1x2的图幅中的第2张图
plt.imshow(imout)
plt.show() # 显示图像