import glob
import time
import cv2
import pytorch_msssim
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from pandas import np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def Myloader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder

    def __getitem__(self, item):
        img= self.data[item]
        label = img.split('\\')[-1].split('_')[2]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def load_Data(train_size, test_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((200, 200)),
        torchvision.transforms.ToTensor()
    ])
    img_name =[]
    img_path = []
    for jpgfile in glob.glob(r'D:\demo\PyPro\TJGradutionProData\cropped_lable\*.jpg'):
        img_path.append(jpgfile)
        img_name.append(jpgfile.split('\\')[-1])
    X_train, X_test, y_train, y_test = train_test_split(img_path, img_name, test_size=0.7, shuffle=True)
    train_data = MyDataset(X_train, transform=transforms, loder=Myloader)
    test_data = MyDataset(X_test, transform=transforms, loder=Myloader)

    train_data = DataLoader(dataset=train_data, batch_size=1, num_workers=0, pin_memory=True)
    test_data = DataLoader(dataset=test_data, batch_size=1, num_workers=0, pin_memory=True)

    return train_data, test_data


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


torch.backends.cudnn.benchmark = True
model = Net().cuda()
print(model)
train_size = 5000
test_size = 200
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()
train_loader, test_loader = load_Data(train_size, test_size)
start = time.time()
for epoch in range(10):
    epoch_start = time.time()
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        out = model(torch.autograd.Variable(batch_x, requires_grad=True))
        loss = loss_func(out, batch_y)
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss, out, batch_x, batch_y
        torch.cuda.empty_cache()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_loader)), train_acc / (len(train_loader))))
    epoch_end = time.time()
    print('Time cost: {}'.format(str(epoch_end - epoch_start)))
    if train_acc / (len(train_loader)) > 0.99:
        break
end = time.time()
print('ALL time cost: {}'.format(str(end - start)))

torch.save(model, 'classification.pkl')

model.eval()
eval_loss = 0.
eval_acc = 0.
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_loader)), eval_acc / (len(test_loader))))
