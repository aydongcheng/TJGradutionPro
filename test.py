import cv2
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

jpgfile = r'D:\demo\PyPro\TJGradutionProData\uncropped\xtest\0.jpg'
img = cv2.cvtColor(cv2.imread(jpgfile), cv2.COLOR_RGB2BGR)
median_img = cv2.medianBlur(img, 5)
gaosi_img = cv2.GaussianBlur(img, (5,5), 11, 11)
avg_img = cv2.blur(img, (5,5))

finish_img = Image.fromarray(median_img)
finish_img.save(r'D:\demo\PyPro\TJGradutionPro\median_img.jpg')
finish_img = Image.fromarray(gaosi_img)
finish_img.save(r'D:\demo\PyPro\TJGradutionPro\gaosi_img.jpg')
finish_img = Image.fromarray(avg_img)
finish_img.save(r'D:\demo\PyPro\TJGradutionPro\avg_img.jpg')
