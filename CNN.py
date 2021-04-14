import time

import cv2
import pytorch_msssim
import torch
import torchvision
import matplotlib.pyplot as plt
from pandas import np
from torchvision import transforms

Xtrain = []
Xtest = []
ytrain = []
ytest = []

for i in range(300):
    Xtrain.append(
        cv2.cvtColor(cv2.imread(r'D:\demo\PyPro\TJGradutionProData\xtrain\{}.jpg'.format(str(i))), cv2.COLOR_RGB2BGR))
    ytrain.append(
        cv2.cvtColor(cv2.imread(r'D:\demo\PyPro\TJGradutionProData\ytrain\{}.jpg'.format(str(i))), cv2.COLOR_RGB2BGR))
for i in range(20):
    Xtest.append(
        cv2.cvtColor(cv2.imread(r'D:\demo\PyPro\TJGradutionProData\xtest\{}.jpg'.format(str(i))), cv2.COLOR_RGB2BGR))
    ytest.append(
        cv2.cvtColor(cv2.imread(r'D:\demo\PyPro\TJGradutionProData\ytest\{}.jpg'.format(str(i))), cv2.COLOR_RGB2BGR))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block_size = 3
        self.block_deep = 3
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1", torch.nn.Sequential(torch.nn.Conv2d(3, 64, (3, 3), 1, 1), torch.nn.ReLU()))
        for d in range(self.block_deep):
            for i in range(self.block_size):
                self.conv.add_module("conv2 " + str(d * self.block_size + i), torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, (3, 3), 1, 1),
                    torch.nn.ReLU()))
        self.conv.add_module("conv3", torch.nn.Sequential(torch.nn.Conv2d(64, 3, (3, 3), 1, 1)))
        # self.conv1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 64, (3, 3), 1, 1),
        #     torch.nn.ReLU())
        # self.conv2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(64, 64, (3, 3), 1, 1),
        #     torch.nn.ReLU())
        # self.conv3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(64, 3, (3, 3), 1, 1)
        # )

    def forward(self, x):
        conv1_out = self.conv[0](x)
        conv2_x = conv1_out
        conv2_out = [conv1_out]
        for d in range(self.block_deep):
            for i in range(self.block_size):
                conv2_out.append(self.conv[i + d * self.block_size + 1](conv2_out[i + d * self.block_size]))
            conv2_out[(d + 1) * self.block_size] += conv2_x
            conv2_x = conv2_out[(d + 1) * self.block_size]
        conv3_out = self.conv[1 + self.block_deep * self.block_size](conv2_out[len(conv2_out) - 1])
        return conv3_out


model = Net()
print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = pytorch_msssim.SSIM()
# loss_func = torch.nn.SmoothL1Loss()
start = time.time()
for epoch in range(10):
    epoch_start = time.time()
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    for i in range(len(Xtrain)):
        batch_x = cv2.medianBlur(Xtrain[i], 3)
        batch_x = torchvision.transforms.ToTensor()(batch_x).unsqueeze(0)
        batch_y = torchvision.transforms.ToTensor()(ytrain[i]).unsqueeze(0)
        out = model(torch.autograd.Variable(batch_x, requires_grad=True))
        optimizer.zero_grad()
        loss = 1 - loss_func(out, batch_y)
        # print(loss.item())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}'.format(train_loss / (len(Xtrain))))
    epoch_end = time.time()
    print('Time cost: {}'.format(str(epoch_end-epoch_start)))
end = time.time()
print('ALL time cost: {}'.format(str(end-start)))

model.eval()
eval_loss = 0.
for i in range(len(Xtest)):
    batch_x = cv2.medianBlur(Xtest[i], 3)
    batch_x = torchvision.transforms.ToTensor()(batch_x).unsqueeze(0)
    batch_y = torchvision.transforms.ToTensor()(ytest[i]).unsqueeze(0)
    out = model(batch_x)
    loss = 1 - loss_func(out, batch_y)
    print(loss.item())
    eval_loss += loss.item()
    out_img = out.squeeze(0).detach().numpy()
    maxValue = out_img.max()
    out_img = out_img * 255 / maxValue
    mat = np.uint8(out_img)
    mat = mat.transpose(1, 2, 0)
    mat = cv2.medianBlur(mat, 5)
    plt.imshow(mat)
    plt.show()
print('Test Loss: {:.6f}'.format(eval_loss / (len(Xtest))))
