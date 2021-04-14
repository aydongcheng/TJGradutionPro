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

for i in range(10):
    Xtrain.append(
        cv2.cvtColor(cv2.imread(r'D:\demo\PyPro\TJGradutionProData\xtrain\{}.jpg'.format(str(i))), cv2.COLOR_RGB2BGR))
    ytrain.append(
        cv2.cvtColor(cv2.imread(r'D:\demo\PyPro\TJGradutionProData\ytrain\{}.jpg'.format(str(i))), cv2.COLOR_RGB2BGR))
for i in range(10):
    Xtest.append(
        cv2.cvtColor(cv2.imread(r'D:\demo\PyPro\TJGradutionProData\xtest\{}.jpg'.format(str(i))), cv2.COLOR_RGB2BGR))
    ytest.append(
        cv2.cvtColor(cv2.imread(r'D:\demo\PyPro\TJGradutionProData\ytest\{}.jpg'.format(str(i))), cv2.COLOR_RGB2BGR))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, (3, 3), 1, 1),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, (3, 3), 1, 1),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, (3, 3), 1, 1)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        return conv3_out


model = Net()
print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = pytorch_msssim.SSIM()
# loss_func = torch.nn.SmoothL1Loss()

for epoch in range(10):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    for i in range(len(Xtrain)):
        batch_x = cv2.medianBlur(Xtrain[i], 5)
        batch_x = torchvision.transforms.ToTensor()(batch_x).unsqueeze(0)
        batch_y = torchvision.transforms.ToTensor()(ytrain[i]).unsqueeze(0)
        out = model(torch.autograd.Variable(batch_x, requires_grad=True))
        optimizer.zero_grad()
        loss = 1-loss_func(out, batch_y)
        print(loss.item())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}'.format(train_loss / (len(Xtrain))))

model.eval()
eval_loss = 0.
for i in range(len(Xtest)):
    batch_x = cv2.medianBlur(Xtest[i], 5)
    batch_x = torchvision.transforms.ToTensor()(batch_x).unsqueeze(0)
    batch_y = torchvision.transforms.ToTensor()(ytest[i]).unsqueeze(0)
    out = model(batch_x)
    loss = 1-loss_func(out, batch_y)
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
