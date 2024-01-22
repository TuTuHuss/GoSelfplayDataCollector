import os
from PIL import Image, ImageFilter
import pickle
from paddleocr import PaddleOCR

# need to run only once to download and load model into memory
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

threshold = 180
pixel_mapping_table = [0] * threshold + [1] * (256 - threshold)


def ocr_from_img(img_path):
    result = ocr.ocr(img_path, cls=True)[0]
    total_text = ''
    if result:
        for idx in range(len(result)):
            total_text += result[idx][1][0]
            print(result[idx])
    return total_text


def preprocess_img(img_path):
    img = Image.open(img_path).convert("L")
    img = img.point(pixel_mapping_table, '1')

    return img


if __name__ == '__main__':
    idx = 0
    total_dataset = []
    while True:
        if not os.path.exists(f'./data/{idx}_expl.png'):
            break

        expl_path = f'./data/{idx}_expl.png'
        board_path = f'./data/{idx}_board.png'

        # Get explanations from expl-image.
        expl = ocr_from_img(expl_path)

        # Preprocess the board image.
        board = preprocess_img(board_path)
        board.save(f'./processed_data/{idx}_board.png')

        total_dataset.append({
            'board': board,
            'expl': expl
        })

        idx += 1

    # Save to the total dataset.
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(total_dataset, f)

