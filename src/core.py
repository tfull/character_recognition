from PIL import Image
import numpy as np
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from .config import Config


class Model(nn.Module):
    def __init__(self, image_size, output):
        super(Model, self).__init__()
        n = ((image_size - 4) // 2 - 4) // 2

        self.conv1 = nn.Conv2d(1, 4, 5)
        self.relu1 = nn.ReLU()
        self.normal1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(4, 16, 5)
        self.relu2 = nn.ReLU()
        self.normal2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n * n * 16, 1024)
        self.relu3 = nn.ReLU()
        self.normal3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(1024, 256)
        self.relu4 = nn.ReLU()
        self.normal4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(256, output)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.normal1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.normal2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.normal3(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.normal4(x)
        x = self.dropout4(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x


def double_range(a1, a2, chunk = 100):
    records = []

    for x1 in a1:
        for x2 in a2:
            records.append((x1, x2))
            if len(records) >= chunk:
                yield records
                records = []

    if len(records) > 0:
        yield records


def train(index, model, device, i_image_list, model_path = None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    n_epoch = 2
    image_chunk = 100

    model.train()

    for i_epoch in range(n_epoch):
        count = 0
        for double_list in double_range(i_image_list, range(len(index["characters"])), chunk = image_chunk):
            inputs = []
            labels = []

            for i_image, i_entity in double_list:
                entity = index["characters"][i_entity]
                inputs.append(read_image(entity["code"], entity["character"], i_image))
                labels.append(i_entity)

            count += len(labels)

            inputs = torch.tensor(inputs).float().to(device)
            labels = torch.tensor(labels).long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sys.stderr.write("epoch: {}, images: {}, loss: {}\n".format(i_epoch + 1, count, loss.item()))

    if model_path is not None:
        torch.save(model.to("cpu").state_dict(), model_path)


def test(index, model, device, i_image_list):
    model.eval()

    image_chunk = 100

    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for double_list in double_range(i_image_list, range(len(index["characters"])), chunk = image_chunk):
            inputs = []
            labels = []

            for i_image, i_entity in double_list:
                entity = index["characters"][i_entity]
                inputs.append(read_image(entity["code"], entity["character"], i_image))
                labels.append(i_entity)

            inputs = torch.tensor(inputs).float().to(device)
            labels = torch.tensor(labels).long().to(device)

            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            judge = labels == prediction

            correct_count += int(judge.sum())
            total_count += len(judge)

    print("Accuracy: {:.2f}%".format(correct_count / total_count * 100))


def read_image(code, character, i_image):
    path_image = Config.data_directory + "/{code}_{character}/{character}_{i_image}.png".format(
        code = code,
        character = character,
        i_image = i_image
    )

    image = Image.open(path_image).convert("L")

    return np.array(image).reshape(1, Config.image_size, Config.image_size) / 255


def read_index():
    with open(Config.data_directory + "/index.yml") as f:
        return yaml.load(f, Loader = yaml.SafeLoader)


def get_device(model):
    if torch.cuda.is_available():
        model.cuda()
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def experiment():
    index = read_index()
    number_of_image = index["number"]

    model = Model(Config.image_size, number_of_image)
    device = get_device(model)

    i_image_list = list(range(1, number_of_image + 1))
    random.shuffle(i_image_list)

    train(index, model, device, i_image_list[:-5])
    test(index, model, device, i_image_list[-5:])


def pretrain():
    index = read_index()
    number_of_image = index["number"]

    model = Model(Config.image_size, number_of_image)
    device = get_device(model)

    i_image_list = list(range(1, number_of_image + 1))
    random.shuffle(i_image_list)

    train(index, model, device, i_image_list, model_path = Config.data_directory + "/model.h5")


if __name__ == '__main__':
    main()
