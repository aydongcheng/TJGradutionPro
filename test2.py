# ConvolutionDenoising.py
from PIL import Image # 导入PIL的Image类
import numpy as np
from pylab import * # 绘图模块

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
            a= im[x-l:x+l+1, y-l:y+l+1]*kernel
            b =sum(im[x-l:x+l+1, y-l:y+l+1]*kernel)
            result[x, y] = sum(im[x-l:x+l+1, y-l:y+l+1]*kernel)
    return result

# 卷积核大小n须为奇数，这里采用均值滤波
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])*1/9

im = Image.open('1_0_0_20161219154724341.jpg').convert('L') # 读取图像并转化为灰度格式
im = np.array(im) # 转化为数组形式
imout = filter2d(im, kernel)

gray() # 灰度格式输出
subplot(1,2,1) # 1x2的图幅中的第1张图
imshow(im)
subplot(1,2,2) # 1x2的图幅中的第2张图
imshow(imout)
show() # 显示图像
