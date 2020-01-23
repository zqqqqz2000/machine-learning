import cv2
import torch
import torch.nn as nn
from typing import *
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

# Hyper Parameters
# the iter times of training
EPOCH = 3
# set the batch size(how many simples use to train the model once)
BATCH_SIZE = 10
# learning rate
LR = 0.001
# if cuda is available, use cuda to up the rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# the structure of the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            # 28 * 28 is the input size
            nn.Linear(28 * 28, 30),
            # active func, and to save the calculate cost, the inplace is True
            nn.ReLU(True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(30, 10)
        )

    def forward(self, image: torch.Tensor):
        x = self.fc1(image)
        out = self.fc2(x)
        return out


# the func to load the mnist dataset
def mnist_loader(download: bool = True) -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    train_data = torchvision.datasets.MNIST(
        root='./data/mnist/',
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=download,
    )
    test_data = torchvision.datasets.MNIST(
        root='./data/mnist/',
        train=False,  # this is test data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=download,
    )
    return train_data, test_data


if __name__ == '__main__':
    # load the dataset
    train_set, test_set = mnist_loader(True)
    # init the model
    net: nn.Module = Net()
    # move to the available device
    net.to(device)
    # the criterion model
    criterion = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = optim.Adam(net.parameters(), lr=LR)
    # init the dataLoader
    train_loader: DataLoader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader: DataLoader = Data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)
    # start train
    for i in range(EPOCH):
        for images, labels in train_loader:
            # because the model is liner nn, so the picture must be flatten as the input
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            predict = net.forward(images)
            # calculate the loss
            loss = criterion(predict, labels)
            # zero the grad
            optimizer.zero_grad()
            # bp
            loss.backward()
            # step to calculate the grad to the model parameters
            optimizer.step()
        # print the training loss each 2 epochs
        if not (i + 1) % 2:
            print(f'loss: {loss.item()}')

    # test
    with torch.no_grad():
        total_loss = 0
        count = 0
        # init the var to avoid the IDE warning
        image: torch.Tensor = None
        label: torch.Tensor = None
        predict: torch.Tensor = None
        for image, label in test_set:
            # flatten and choose fit device
            image_ = image.view(-1, 28 * 28).to(device)
            predict: torch.Tensor = net(image_)
            loss = criterion(predict, torch.tensor([label]))
            total_loss += loss.item()
            count += 1
        total_loss /= count
        print(f'the final loss is: {total_loss}')
        # show the result, the predict label will show on the console, and the picture will show as the cv2 window
        for image, label in test_set:
            image_ = image.view(-1, 28 * 28).to(device)
            predict: torch.Tensor = net(image_)
            loss = criterion(predict, torch.tensor([label]))
            total_loss += loss.item()
            count += 1
            img = image[0]
            label = predict.argmax()
            print(f'predict: {label}')
            cv2.imshow('img', img.numpy())
            cv2.waitKey(0)
