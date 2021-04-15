import time

import cv2
import pytorch_msssim
import torch
import torchvision
import matplotlib.pyplot as plt
from pandas import np
from torch.utils.data import Dataset, DataLoader


def Myloader(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)


def init_process(Xpath, ypath, len):
    data = []
    for i in range(len):
        data.append([Xpath % i, ypath % i])
    return data


class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder

    def __getitem__(self, item):
        noise_img, clean_img = self.data[item]
        noise_img = self.loader(noise_img)
        # noise_img = cv2.medianBlur(noise_img, 3)
        noise_img = self.transform(noise_img).unsqueeze(0)
        clean_img = self.loader(clean_img)
        clean_img = self.transform(clean_img).unsqueeze(0)
        return noise_img, clean_img

    def __len__(self):
        return len(self.data)


def load_Data(train_size, test_size):
    transforms = torchvision.transforms.ToTensor()
    root = r'D:\demo\PyPro\TJGradutionProData'
    train_data = init_process(root + r'\xtrain\%d.jpg', root + r'\ytrain\%d.jpg', train_size)
    train_data = MyDataset(train_data, transform=transforms, loder=Myloader)
    test_data = init_process(root + r'\xtest\%d.jpg', root + r'\ytest\%d.jpg', test_size)
    test_data = MyDataset(test_data, transform=transforms, loder=Myloader)

    train_data = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    test_data = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    return train_data, test_data


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


torch.backends.cudnn.benchmark = True
model = Net()
print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = pytorch_msssim.SSIM()
# loss_func = torch.nn.SmoothL1Loss()
train_loader, test_loader = load_Data(50, 10)
start = time.time()
for epoch in range(10):
    epoch_start = time.time()
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    for batch_x, batch_y in train_loader:
        out = model(torch.autograd.Variable(batch_x[0], requires_grad=True))
        loss = 1 - loss_func(out, batch_y[0])
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss, out
        torch.cuda.empty_cache()
    print('Train Loss: {:.6f}'.format(train_loss / (len(train_loader))))
    epoch_end = time.time()
    print('Time cost: {}'.format(str(epoch_end - epoch_start)))
end = time.time()
print('ALL time cost: {}'.format(str(end - start)))

model.eval()
eval_loss = 0.
for batch_x, batch_y in test_loader:
    out = model(batch_x[0])
    loss = 1 - loss_func(out, batch_y[0])
    print(loss.item())
    eval_loss += loss.item()
    out_img = out.squeeze(0).detach().numpy()
    maxValue = out_img.max()
    out_img = out_img * 255 / maxValue
    mat = np.uint8(out_img)
    mat = mat.transpose(1, 2, 0)
    plt.imshow(mat)
    plt.show()
print('Test Loss: {:.6f}'.format(eval_loss / (len(test_loader))))
