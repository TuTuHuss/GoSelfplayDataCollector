import os
import pickle

from img2txt import board2txt, ocr


def ocr_from_img(img_path):
    result = ocr.ocr(img_path, cls=True)[0]
    total_text = ''
    if result:
        for idx in range(len(result)):
            total_text += result[idx][1][0]
    return total_text


if __name__ == '__main__':
    idx = 0
    total_dataset = []
    while True:
        if not os.path.exists(f'./data/{idx}_expl.png'):
            break
        print(f"Processing index: {idx}...")

        expl_path = f'./data/{idx}_expl.png'
        board_path = f'./data/{idx}_board.png'

        # Get explanations from expl-image.
        expl = ocr_from_img(expl_path)

        # Get txt-formed board.
        board = board2txt(board_path)

        prompt = f'The following is a game board of Go. ' \
                 f'The # on the board is black stone, the o on the board is white stone and ' \
                 f'the . on the board means that there is no stone. ' \
                 f'Tokens between two parenthesis is related to the explanations about this board state. ' \
                 f'Here is the board state: \n <\\board> {board} \n <\\board>. ' \
                 f'Please generate the corresponding explanations about this board state.'

        total_dataset.append({
            'sentence': prompt,
            'answer': expl
        })

        idx += 1

    # Save to the total dataset.
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(total_dataset, f)
