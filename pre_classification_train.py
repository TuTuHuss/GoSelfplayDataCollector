import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import torchvision


import os
import pandas as pd
import random

dataset = []


def preprocess_dataset(paths):
    for path in paths:
        preprocess_one_image(path)
    global dataset
    random.shuffle(dataset)


def preprocess_one_image(path):
    """
    Generate classification dataset from the raw annotations.
    """
    xlsx_path = os.path.join(path, 'label.xlsx')
    print(xlsx_path)
    data = pd.read_excel(xlsx_path, dtype=str, keep_default_na=False)

    types = data['Type']
    shapes = data['Shape']
    for i in range(len(types)):
        typ = types[i]
        sha = shapes[i]

        if typ == 'None' and sha == 'None':
            content = 'empty'
        elif typ == 'None' and sha.isalpha():
            content = 'char'
        elif typ.lower() in ['black', 'white'] and sha == 'None':
            content = 'pure_stone'
        elif typ.lower() in ['black', 'white'] and sha != 'None':
            if sha.lower() == 'triangle':
                content = 'triangle'
            elif sha.lower() == 'square':
                content = 'square'
            elif sha.isdigit() and typ.lower() == 'black':
                content = 'digit_b'
            elif sha.isdigit() and typ.lower() == 'white':
                content = 'digit_w'
            else:
                raise ValueError(f'Bad combination of label: {typ} and {sha} when processing {xlsx_path}')
        else:
            raise ValueError(f'Bad combination of label: {typ} and {sha} when processing {xlsx_path}')

        if content == '十':
            rand = random.random()
            if rand < 0.9:
                continue
        dataset.append({
            'type': content,
            'img_path': os.path.join(path, f"patch_{i}.png")
        })


class MyDataset(Dataset):
    def __init__(self, data):
        self.dataset = data
        self.class_dict = {
            'empty': 0,
            'char': 1,
            'pure_stone': 2,
            'triangle': 3,
            'square': 4,
            'digit_w': 5,
            'digit_b': 6
        }
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        img_path, cls = self.dataset[item]['img_path'], self.dataset[item]['type']
        img = Image.open(img_path)
        cls = self.class_dict[cls]
        return self.transform(img), cls

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    preprocess_dataset([f'stone/img{i}' for i in range(1, 11)])
    val_frac = 0.2
    val_num = int(len(dataset) * val_frac)

    train_dataset = MyDataset(dataset[:-val_num])
    val_dataset = MyDataset(dataset[-val_num:])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

    model = mobilenet_v3_small(weight=torchvision.models.MobileNet_V3_Small_Weights, num_classes=7).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(50):
        model.train()
        for _, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            o = model(batch_x)
            loss = criterion(o, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            tot_acc, tot_loss = [], []
            for _, (batch_x, batch_y) in enumerate(val_dataloader):
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                o = model(batch_x)
                loss = criterion(o, batch_y)

                pred_o = torch.argmax(o, dim=-1)
                acc = torch.mean((pred_o == batch_y).float())
                tot_acc.append(acc)
                tot_loss.append(loss)
        print(f"Epoch: {epoch}, Val loss: {sum(tot_loss) / len(tot_loss)}, Val acc: {sum(tot_acc) / len(tot_acc)}.")
    torch.save(model.state_dict(), 'classification_model.pth.tar')
