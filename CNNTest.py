import torch
import cv2
from PIL import Image
from pandas import np
from torch.utils.data import Dataset, DataLoader
import torchvision


def Myloader(path):
    return Image.open(path).convert('RGB')


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
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((200, 200)),
        torchvision.transforms.ToTensor()
    ])
    root = r'D:\demo\PyPro\TJGradutionProData'
    train_data = init_process(root + r'\xtrain\%d.jpg', root + r'\ytrain\%d.jpg', train_size)
    train_data = MyDataset(train_data, transform=transforms, loder=Myloader)
    test_data = init_process(root + r'\xtest\%d.jpg', root + r'\ytest\%d.jpg', test_size)
    test_data = MyDataset(test_data, transform=transforms, loder=Myloader)

    train_data = DataLoader(dataset=train_data, batch_size=50, num_workers=0, pin_memory=True)
    test_data = DataLoader(dataset=test_data, batch_size=1, num_workers=0, pin_memory=True)

    return train_data, test_data


def load_Data(train_size, test_size):
    transforms = torchvision.transforms.ToTensor()
    root = r'D:\demo\PyPro\TJGradutionProData'
    train_data = init_process(root + r'\xtrain\%d.jpg', root + r'\ytrain\%d.jpg', train_size)
    train_data = MyDataset(train_data, transform=transforms, loder=Myloader)
    test_data = init_process(root + r'\xtest\%d.jpg', root + r'\ytest\%d.jpg', test_size)
    test_data = MyDataset(test_data, transform=transforms, loder=Myloader)

    train_data = DataLoader(dataset=train_data, batch_size=1, num_workers=0, pin_memory=True)
    test_data = DataLoader(dataset=test_data, batch_size=1, num_workers=0, pin_memory=True)

    return train_data, test_data


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


train_loader, test_loader = load_Data(5000, 200)
loss_func = torch.nn.L1Loss()
model = torch.load('cnn-bn10.pkl')
model.eval()
eval_loss = 0.
count = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        out = model(batch_x[0])
        loss = loss_func(out, batch_y[0])
        eval_loss += loss.item()
        out_img = out.cpu().squeeze(0).detach().numpy()
        maxValue = out_img.max()
        out_img = out_img * 255 / maxValue
        mat = np.uint8(out_img)
        mat = mat.transpose(1, 2, 0)
        # plt.imshow(mat)
        # plt.show()
        mat = Image.fromarray(mat)
        mat.save(r'D:\demo\PyPro\TJGradutionProData\testResult-bn10\{}.jpg'.format(str(count)))
        count += 1
    print('Test Loss: {:.6f}'.format(eval_loss / (len(test_loader))))
