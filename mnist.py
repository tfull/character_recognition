import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import torch.optim as optim

from config import Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(12 * 12 * 32, 128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x


def main():
    n_epoch = 2

    download_flag = not os.path.exists(Config.data_directory + "/mnist")

    mnist_train = torchvision.datasets.MNIST(
        Config.data_directory + "/mnist",
        train = True,
        download = download_flag,
        transform = torchvision.transforms.ToTensor()
    )

    mnist_test = torchvision.datasets.MNIST(
        Config.data_directory + "/mnist",
        train = False,
        download = download_flag,
        transform = torchvision.transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(mnist_train,  batch_size = 100,  shuffle = True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 1000, shuffle = False)

    model = Model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for i_epoch in range(n_epoch):
        for i_batch, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print("epoch: {}, train: {}, loss: {}".format(i_epoch + 1, i_batch + 1, loss.item()))

    correct_count = 0
    record_count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            judge = prediction == labels
            correct_count += int(judge.sum())
            record_count += len(judge)

    print("Accuracy: {:.2f}".format(correct_count / record_count * 100))


if __name__ == '__main__':
    main()
