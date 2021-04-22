import glob

import torch
import cv2
from PIL import Image
from pandas import np
from torch.utils.data import Dataset, DataLoader
import torchvision


def Myloader(path):
    return Image.open(path).convert('RGB')


noise_path = []
for jpgfile in glob.glob(r'D:\demo\PyPro\TJGradutionProData\cropped_lable\*.jpg'):
    noise_path.append(jpgfile)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block_size = 3
        self.block_deep = 5
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1", torch.nn.Sequential(torch.nn.Conv2d(3, 64, (3, 3), 1, 1), torch.nn.ReLU()))
        for d in range(self.block_deep):
            for i in range(self.block_size):
                self.conv.add_module("conv2 " + str(d * self.block_size + i), torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, (3, 3), 1, 1),
                    torch.nn.ReLU()))
        self.conv.add_module("conv3", torch.nn.Sequential(torch.nn.Conv2d(64, 3, (3, 3), 1, 1)))

    def forward(self, x):
        conv1_out = self.conv[0](x)
        conv2_out = conv1_out
        for d in range(self.block_deep):
            conv2_x = conv2_out
            for i in range(self.block_size):
                conv2_out = self.conv[i + d * self.block_size + 1](conv2_out)
            conv2_out += conv2_x
        conv3_out = self.conv[1 + self.block_deep * self.block_size](conv2_out)
        return conv3_out


model = torch.load('cnn-bn10.pkl')
model.eval()
with torch.no_grad():
    for path in noise_path:
        img_name = path.split('\\')[-1]
        batch_x = torchvision.transforms.ToTensor()(Myloader(path)).unsqueeze(0)
        batch_x = batch_x.cuda()
        out = model(batch_x)
        out_img = out.cpu().squeeze(0).detach().numpy()
        maxValue = out_img.max()
        out_img = out_img * 255 / maxValue
        mat = np.uint8(out_img)
        mat = mat.transpose(1, 2, 0)
        # plt.imshow(mat)
        # plt.show()
        mat = Image.fromarray(mat)
        mat.save(r'D:\demo\PyPro\TJGradutionProData\testResult-bn10\{}'.format(img_name))
