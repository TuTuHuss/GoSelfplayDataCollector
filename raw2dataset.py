import os
import pandas as pd
import random
import shutil

dataset = []


def preprocess_dataset(paths):
    for path in paths:
        preprocess_one_image(path)
    global dataset
    random.shuffle(dataset)


def preprocess_one_image(path):
    xlsx_path = os.path.join(path, 'label.xlsx')
    print(xlsx_path)
    data = pd.read_excel(xlsx_path, dtype=str, keep_default_na=False)

    types = data['Type']
    shapes = data['Shape']
    for i in range(len(types)):
        typ = types[i]
        sha = shapes[i]

        if typ == 'None' and sha == 'None':
            content = '十'
        elif typ == 'None' and sha.isalpha():
            content = sha
        elif typ.lower() in ['black', 'white'] and sha == 'None':
            content = '⚪'
        elif typ.lower() in ['black', 'white'] and sha != 'None':
            if sha.lower() == 'triangle':
                content = '△'
            elif sha.lower() == 'square':
                content = '□'
            elif sha.isdigit():
                content = sha
            else:
                raise ValueError(f'Bad combination of label: {typ} and {sha} when processing {xlsx_path}')
        else:
            raise ValueError(f'Bad combination of label: {typ} and {sha} when processing {xlsx_path}')

        if content == '十':
            rand = random.random()
            if rand < 0.9:
                continue
        dataset.append({
            'content': content,
            'img_path': os.path.join(path, f"patch_{i}.png")
        })


def generate_dataset(val_frac=0.2):
    val_num = int(len(dataset) * val_frac)
    val_data = dataset[-val_num:]
    train_data = dataset[:-val_num]

    if not os.path.exists('train_data'):
        os.mkdir('train_data')
        os.mkdir('train_data/train')
        os.mkdir('train_data/test')

    train_str = ''
    for idx, item in enumerate(train_data):
        img_path, content = item['img_path'], item['content']
        new_path = f'train_data/train/word_{idx}.png'
        shutil.copy(img_path, new_path)
        label_path = f'train/word_{idx}.png'
        train_str += f"{label_path} {content}\n"

    test_str = ''
    for idx, item in enumerate(val_data):
        img_path, content = item['img_path'], item['content']
        new_path = f'train_data/test/word_{idx}.png'
        shutil.copy(img_path, new_path)
        label_path = f'test/word_{idx}.png'
        test_str += f"{label_path} {content}\n"

    with open('train_data/train_list.txt', 'w') as f:
        f.write(train_str)
    with open('train_data/val_list.txt', 'w') as f:
        f.write(test_str)


if __name__ == '__main__':
    preprocess_dataset([f'stone/img{i}' for i in range(1, 11)])
    generate_dataset(val_frac=0.2)
    print(len(dataset))
