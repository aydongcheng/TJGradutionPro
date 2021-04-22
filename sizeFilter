import cv2
import glob

from PIL import Image

for jpgfile in glob.glob(r'D:\demo\PyPro\TJGradutionProData\img\*.jpg'):
    img = cv2.cvtColor(cv2.imread(jpgfile), cv2.COLOR_RGB2BGR)
    if 40000 < img.shape[0]*img.shape[1]< 1500000:
        img_name = jpgfile.split('\\')[-1]
        img = Image.fromarray(img)
        img.save(r'D:\demo\PyPro\TJGradutionProData\img-filter\{0}'.format(img_name))