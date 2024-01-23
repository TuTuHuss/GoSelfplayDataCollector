import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from img2txt import get_txt_pred

import os
import pandas as pd
import random

dataset = []


def preprocess_dataset(paths):
    for path in paths:
        preprocess_one_image(path)
    global dataset
    random.shuffle(dataset)


def load_total_patches():
    total_paths = [item['img_path'] for item in dataset]
    return [cv2.imread(pth) for pth in total_paths]


def post_process(raw_ocr):
    for j in range(len(raw_ocr)):
        if isinstance(raw_ocr[j], str):
            dict1 = {'①': "1", '②': "2", '③': "3", '④': "4", '⑤': "5"}
            raw_ocr[j] = raw_ocr[j].upper()
            func1 = lambda c: c if c not in dict1 else dict1[c]
            func = lambda c: c in "0123456789ABCDEFG△□"

            raw_ocr[j] = ''.join(map(func1, raw_ocr[j]))
            raw_ocr[j] = ''.join(filter(func, raw_ocr[j]))
            if len(raw_ocr[j]) == 0:
                raw_ocr[j] = None
    return raw_ocr


def batch_ocr(patches):
    PADDED_IMG_SIZE = 64
    ret = []
    for pat in patches:
        before_x = (PADDED_IMG_SIZE - pat.shape[0]) // 2
        before_y = (PADDED_IMG_SIZE - pat.shape[0]) // 2

        margin_ratio = 0.10

        pat = pat[int(pat.shape[0] * margin_ratio): int(pat.shape[0] * (1 - margin_ratio)),
              int(pat.shape[1] * margin_ratio): int(pat.shape[1] * (1 - margin_ratio)), :]

        padded_img = np.pad(pat,
                            ((before_x, PADDED_IMG_SIZE - before_x - pat.shape[0]),
                             (before_y, PADDED_IMG_SIZE - before_y - pat.shape[1]),
                             (0, 0)))

        res = ocr.ocr(img=padded_img, det=True)[0]
        if res is None:
            ret.append(None)
        else:
            ret.append(res[0][1][0])

    return post_process(ret)


def get_prediction(ocr_info, cls_info, patches):
    res = copy.deepcopy(ocr_info)
    for i in range(len(ocr_info)):
        x_ocr = ocr_info[i]
        x_cls = cls_info[i]
        x_pat = patches[i]

        res[i] = get_txt_pred(x_ocr, x_cls, x_pat)
    return res


def get_total_ground_truth():
    return [item['desc'] for item in dataset], [item['desc_type'] for item in dataset]


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
            desc = '.'
            desc_type = 0
        elif typ == 'None' and sha.isalpha():
            content = 'char'
            desc = f'.({sha})'
            desc_type = 1
        elif typ.lower() in ['black', 'white'] and sha == 'None':
            content = 'pure_stone'
            stone = '#' if typ.lower() == 'black' else 'o'
            desc = f'{stone}'
            desc_type = 2
        elif typ.lower() in ['black', 'white'] and sha != 'None':
            stone = '#' if typ.lower() == 'black' else 'o'
            if sha.lower() == 'triangle':
                content = 'triangle'
                desc = f'{stone}(△)'
                desc_type = 3
            elif sha.lower() == 'square':
                content = 'square'
                desc = f'{stone}(□)'
                desc_type = 4
            elif sha.isdigit() and typ.lower() == 'black':
                content = 'digit_b'
                desc = f'{stone}({sha})'
                desc_type = 6
            elif sha.isdigit() and typ.lower() == 'white':
                content = 'digit_w'
                desc = f'{stone}({sha})'
                desc_type = 5
            else:
                raise ValueError(f'Bad combination of label: {typ} and {sha} when processing {xlsx_path}')
        else:
            raise ValueError(f'Bad combination of label: {typ} and {sha} when processing {xlsx_path}')

        dataset.append({
            'type': content,
            'img_path': os.path.join(path, f"patch_{i}.png"),
            'desc': desc,
            'desc_type': desc_type
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
    preprocess_dataset([f'stone/img{i}' for i in range(1, 12)])

    # Get cls info
    train_dataset = MyDataset(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = mobilenet_v3_small(num_classes=7)
    sd = torch.load('classification_model.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        tot_res = []
        for _, (batch_x, batch_y) in enumerate(train_dataloader):
            o = model(batch_x)
            pred_o = torch.argmax(o, dim=-1)
            tot_res.append(pred_o)

    tot_res = torch.cat(tot_res, dim=0)
    tot_cls = []

    class_dict = {
        0: 'empty',
        1: 'char',
        2: 'pure_stone',
        3: 'triangle',
        4: 'square',
        5: 'digit_w',
        6: 'digit_b'
    }

    for i in range(tot_res.shape[0]):
        tot_cls.append(class_dict[tot_res[i].item()])

    # Get total patches
    tot_pat = load_total_patches()

    # Get total ocr info
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    tot_ocr = batch_ocr(tot_pat)

    # Get final prediction
    tot_pred = get_prediction(tot_ocr, tot_cls, tot_pat)
    tot_gt, gt_type = get_total_ground_truth()

    # Calculate acc
    cnt = {i: 0 for i in range(7)}
    err = {i: 0 for i in range(7)}
    acc = {i: 0 for i in range(7)}

    err_info = []

    for i in range(len(tot_pred)):
        if tot_pred[i] == tot_gt[i]:
            cnt[gt_type[i]] += 1
        else:
            err[gt_type[i]] += 1
            err_info.append({
                'Ground_truth': tot_gt[i],
                'Class_pred': tot_cls[i],
                'OCR_pred': tot_ocr[i],
                'Final_pred': tot_pred[i],
                'Image': tot_pat[i]
            })
    for i in range(7):
        acc[i] = cnt[i] / (cnt[i] + err[i] + 1)
    print(f'Final Acc: {acc}')
    print(f'Final Right: {cnt}')
    print(f'Final Error: {err}')

    for i, item in enumerate(err_info):
        img = item.pop('Image')
        cv2.imwrite(f'error_info/{i}.png', img)
        with open(f'error_info/{i}.txt', 'w') as f:
            f.write(str(item) + '\n')
