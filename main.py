from PIL import Image
import numpy as np
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config


class Model(nn.Module):
    def __init__(self, output):
        super(Model, self).__init__()
        size = 256
        n = ((size - 4) // 2 - 4) // 2

        self.conv1 = nn.Conv2d(1, 16, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(16, 64, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n * n * 64, 1024)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(1024, 256)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(256, output)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x


def main():
    index = read_index()
    number_of_image = index["number"]

    model = Model(number_of_image)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    i_image_list = list(range(1, number_of_image + 1))
    random.shuffle(i_image_list)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    n_epoch = 2

    model.train()

    for i_epoch in range(n_epoch):
        for i_image in i_image_list[:3]:
            inputs = []
            labels = []

            for i_entity, entity in enumerate(index["characters"]):
                inputs.append(read_image(entity["code"], entity["character"], i_image))
                labels.append(i_entity)

            inputs = torch.tensor(inputs).float().to(device)
            labels = torch.tensor(labels).long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print("i_epoch: {}, i_image: {}, loss: {}".format(i_epoch + 1, i_image, loss.item()))

    model.eval()
    
    inputs = []
    labels = []

    for i_image in i_image_list[-5:]:
        for i_entity, entity in enumerate(index["characters"]):
            inputs.append(read_image(entity["code"], entity["character"], i_image))
            labels.append(i_entity)

    inputs = torch.tensor(inputs).float().to(device)
    labels = torch.tensor(labels).long().to(device)

    outputs = model(inputs)
    _, prediction = torch.max(outputs.data, 1)
    judge = labels == prediction

    print("Accuracy: {:.2f}%".format(int(judge.sum()) / len(judge) * 100))


def read_image(code, character, i_image):
    path_image = Config.data_directory + "/{code}_{character}/{character}_{i_image}.png".format(
        code = code,
        character = character,
        i_image = i_image
    )

    image = Image.open(path_image).convert("L")

    return np.array(image).reshape(1, Config.image_size, Config.image_size)


def read_index():
    with open(Config.data_directory + "/index.yml") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


if __name__ == '__main__':
    main()
