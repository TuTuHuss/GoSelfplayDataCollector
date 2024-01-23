import copy
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="ch")
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)


def cut_img(img: np.ndarray, margin: int = 2):
    """
        Cut off the white margin of the original image. ``margin`` refers to the size of white margin that is not cut.
    """
    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_new = cv2.threshold(img_new, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    edges_y, edges_x = np.where(img_new == 0)
    bottom = max(min(edges_y) - margin, 0)
    top = max(edges_y)
    left = max(min(edges_x) - margin, 0)
    right = max(edges_x)
    height = top - bottom + 2 * margin
    width = right - left + 2 * margin
    res_image = img[bottom:bottom + height, left:left + width]

    return res_image


def get_nearest(from_px: float, to_px: float, interval: float):
    """
        Assume ``to_px`` is close to ``from_px`` + ``i`` * ``interval``. Calculate the closest ``i`` * ``interval`` for
    all integer ``i``. Note that the distance here is measured by Euclidian distance.
    """
    nearest_dist = abs(from_px - to_px)
    res = from_px

    while from_px >= to_px:
        from_px -= interval
        if abs(from_px - to_px) < nearest_dist:
            nearest_dist = (from_px - to_px) ** 2
            res = from_px

    while from_px <= to_px:
        from_px += interval
        if abs(from_px - to_px) < nearest_dist:
            nearest_dist = (from_px - to_px) ** 2
            res = from_px

    return res


def get_estimated_center(orig_est_x: float, orig_est_y: float, points: List, radius: float, momentum=0.99):
    """
        Given the original estimated x and y, get a refined coordination which is aligned to the centers of
    detected circles.
    """
    total_distance = 0
    est_x, est_y = 0, 0
    for p in points:
        px, py = p
        dist = 1 / (abs(px - orig_est_x) + abs(py - orig_est_y))
        est_x += dist * get_nearest(from_px=px, to_px=orig_est_x, interval=2 * radius)
        est_y += dist * get_nearest(from_px=py, to_px=orig_est_y, interval=2 * radius)
        total_distance += dist
    res_x = momentum * orig_est_x + (1 - momentum) * (est_x / total_distance)
    res_y = momentum * orig_est_y + (1 - momentum) * (est_y / total_distance)
    return int(res_x), int(res_y)


def estimate_radius(circles: List) -> float:
    """
        Estimate the radius of stones using the detected circles.
    """
    rounded_circles = np.uint16(np.around(circles))
    rounded_circles = rounded_circles.tolist()
    rounded_circles.sort(key=lambda x: x[2])
    radius_tot = [cir[2] for cir in rounded_circles]
    radius = max(radius_tot, key=radius_tot.count)

    recomputed_radius = []
    for cir in circles:
        if abs(cir[2] - radius) <= 2:
            recomputed_radius.append(cir[2])
    radius = sum(recomputed_radius) / len(recomputed_radius)
    return radius


def patchify(checkerboard: np.ndarray) -> List:
    """
        Patchify the board image. Return a list of small image patches that separately contains a crossing of the board.
    """
    gray = cv2.cvtColor(checkerboard, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    max_radius = 30  # TODO: improvements?

    # Detect circles in the input image.
    circles = cv2.HoughCircles(
        image=gauss,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=100,
        param2=30,
        maxRadius=max_radius
    )[0]

    # Estimate radius using the detected circles.
    est_radius = estimate_radius(circles)

    # Search an optimal estimated board width using the centers of the detected circles.
    min_board_width = 2 * est_radius - 5
    max_board_width = 2 * est_radius + 5
    x_min = min([circles[i][1] for i in range(len(circles))])
    y_min = min([circles[i][0] for i in range(len(circles))])

    best_board_width = -1
    best_error = 100000
    num_candidates = 1000
    bw = min_board_width
    interval = (max_board_width - min_board_width) / num_candidates
    while bw < max_board_width:
        cur_error = 0
        for cir in circles:
            cur_error += get_nearest(x_min, cir[1], bw)
            cur_error += get_nearest(y_min, cir[0], bw)
        if cur_error < best_error:
            best_error = cur_error
            best_board_width = bw
        bw += interval

    # Calculate a better starting point using all circles.
    estimated_points = []
    started_points = []
    for cir in circles:
        x_tmp, y_tmp = cir[1], cir[0]
        if abs(est_radius - cir[2]) < 2:
            estimated_points.append([x_tmp, y_tmp])
            while x_tmp - best_board_width > 0:
                x_tmp -= best_board_width

            while y_tmp - best_board_width > 0:
                y_tmp -= best_board_width

            started_points.append([x_tmp, y_tmp])
    start_x = sum([started_points[i][0] for i in range(len(started_points))]) / len(started_points)
    start_y = sum([started_points[i][1] for i in range(len(started_points))]) / len(started_points)

    # Patchify the image and visualize.
    stone = checkerboard.copy()
    patches = []
    for i in range(19):
        patches.append([])
        for j in range(19):
            # Get the roughly estimated coordination.
            x_cent, y_cent = start_x + i * best_board_width, start_y + j * best_board_width
            # Get the refined coordination.
            x_cent, y_cent = get_estimated_center(x_cent, y_cent, estimated_points, best_board_width / 2)
            if x_cent >= stone.shape[0] + 3 or y_cent >= stone.shape[1] + 3:
                continue
            radius = int(best_board_width / 2) + 1
            # cv2.circle(stone, (y_cent, x_cent), radius, (0, 0, 255), 3)

            x_left, x_right = max(0, x_cent - radius), min(stone.shape[0], x_cent + radius)
            y_left, y_right = max(0, y_cent - radius), min(stone.shape[1], y_cent + radius)
            patches[-1].append(checkerboard[x_left: x_right, y_left: y_right, :])

    # for cir in circles:
    #     cv2.circle(stone, (int(cir[0]), int(cir[1])), int(cir[2]), (0, 255, 0), 3)
    # plt.figure(figsize=(10, 10), dpi=80)
    # plt.imshow(stone)
    # plt.show()

    return patches


def load_img(path: str):
    src = cv2.imread(path)
    src = cut_img(src)
    return src


def batch_ocr(patches):
    PADDED_IMG_SIZE = 64
    ret = []
    for line in patches:
        ret.append([])
        for pat in line:
            # input()
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
                ret[-1].append(None)
            else:
                ret[-1].append(res[0][1][0])
    return ret


def _post_process(s):
    dict1 = {'①': "1", '②': "2", '③': "3", '④': "4", '⑤': "5"}
    s = s.upper()
    func1 = lambda c: c if c not in dict1 else dict1[c]
    func = lambda c: c in "0123456789ABCDEFG△□"
    s = ''.join(map(func1, s))
    s = ''.join(filter(func, s))
    if len(s) == 0:
        s = None
    return s


def post_process(raw_ocr):
    for j in range(len(raw_ocr)):
        for i in range(len(raw_ocr[j])):
            if isinstance(raw_ocr[j][i], str):
                raw_ocr[j][i] = _post_process(raw_ocr[j][i])



def batch_classification(patches):
    import torchvision.transforms
    from torchvision.models import mobilenet_v3_small
    import torch
    from PIL import Image
    transform = torchvision.transforms.Compose([
        Image.fromarray,
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
    ])
    class_dict = {
        0: 'empty',
        1: 'char',
        2: 'pure_stone',
        3: 'triangle',
        4: 'square',
        5: 'digit_w',
        6: 'digit_b'
    }

    batch = []
    for line in patches:
        for pp in line:
            batch.append(transform(pp))
    batch = torch.stack(batch, dim=0)

    model = mobilenet_v3_small(weight=torchvision.models.MobileNet_V3_Small_Weights, num_classes=7).eval()
    sd = torch.load('classification_model.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(sd)
    o = model(batch)
    y_pred = torch.argmax(o, dim=-1)

    res = []
    cnt = 0
    for i in range(len(patches)):
        res.append([])
        for j in range(len(patches[i])):
            cls_pred = y_pred[cnt].item()
            res[-1].append(class_dict[cls_pred])
            cnt += 1
    return res


def get_white_ratio(pat):
    margin_ratio = 0.25
    pat = pat[int(pat.shape[0] * margin_ratio): int(pat.shape[0] * (1 - margin_ratio)),
          int(pat.shape[1] * margin_ratio): int(pat.shape[1] * (1 - margin_ratio)), :]

    return np.sum(pat) / np.sum(255 * np.ones_like(pat))


def get_txt_pred(x_ocr, x_cls, x_pat):
    if x_ocr == '△':
        x_cls = 'triangle'
    elif x_ocr == '□':
        x_cls = 'square'

    if x_cls == 'digit_w' or x_cls == 'digit_b':
        inversed_pat = 255 - x_pat
        PADDED_IMG_SIZE = 64
        before_x = (PADDED_IMG_SIZE - inversed_pat.shape[0]) // 2
        before_y = (PADDED_IMG_SIZE - inversed_pat.shape[0]) // 2

        margin_ratio = 0.10
        inversed_pat = inversed_pat[int(inversed_pat.shape[0] * margin_ratio): int(inversed_pat.shape[0] * (1 - margin_ratio)),
              int(inversed_pat.shape[1] * margin_ratio): int(inversed_pat.shape[1] * (1 - margin_ratio)), :]

        padded_img = np.pad(inversed_pat,
                            ((before_x, PADDED_IMG_SIZE - before_x - inversed_pat.shape[0]),
                             (before_y, PADDED_IMG_SIZE - before_y - inversed_pat.shape[1]),
                             (0, 0)))

        res = ocr.ocr(img=padded_img, det=True)[0]
        if res is not None:
            if x_ocr is None:
                x_ocr = _post_process(res[0][1][0])

    white_ratio = get_white_ratio(x_pat)
    if white_ratio > 1 - white_ratio:
        stone_base = 'o'
    else:
        stone_base = '#'

    if x_cls == 'empty':
        ret = f'.'

    elif x_cls == 'char' and x_ocr is None:
        ret = f'.'
    elif x_cls == 'char' and x_ocr in 'ABCDE':
        ret = f'.({x_ocr})'
    elif x_cls == 'char' and x_ocr.isdigit():
        ret = f'{stone_base}({x_ocr})'

    elif x_cls == 'digit_w' and x_ocr is None:
        ret = f'o'
    elif x_cls == 'digit_w' and x_ocr.isdigit():
        ret = f'o({x_ocr})'
    elif x_cls == 'digit_w' and x_ocr in 'ABCDE':
        ret = f'.({x_ocr})'
    elif x_cls == 'digit_w':
        ret = f'o'

    elif x_cls == 'digit_b' and x_ocr is None:
        ret = f'#'
    elif x_cls == 'digit_b' and x_ocr.isdigit():
        ret = f'#({x_ocr})'
    elif x_cls == 'digit_b' and x_ocr in 'ABCDE':
        ret = f'.({x_ocr})'
    elif x_cls == 'digit_b':
        ret = f'#'

    elif x_cls == 'square':
        ret = f'{stone_base}(□)'

    elif x_cls == 'triangle':
        ret = f'{stone_base}(△)'

    elif x_cls == 'pure_stone':
        ret = f'{stone_base}'
    else:
        raise ValueError(f'Bad combination of x_cls: {x_cls} and x_ocr: {x_ocr}')
    return ret


def get_prediction(ocr_info, cls_info, patches):
    res = copy.deepcopy(ocr_info)
    for i in range(len(ocr_info)):
        cols = len(ocr_info[i])
        for j in range(cols):
            x_ocr = ocr_info[i][j]
            x_cls = cls_info[i][j]
            x_pat = patches[i][j]
            res[i][j] = get_txt_pred(x_ocr, x_cls, x_pat)
    return res


def board2txt(path):
    image = load_img(path)
    segmented_patches = patchify(image)

    # OCR
    ocr_res = batch_ocr(segmented_patches)
    post_process(ocr_res)

    # Classification
    cls_res = batch_classification(segmented_patches)
    final_res = get_prediction(ocr_res, cls_res, segmented_patches)

    txt = [' '.join(final_res[i]) for i in range(len(final_res))]
    return '\n'.join(txt)


if __name__ == '__main__':
    ocr_txt = board2txt("./data/1_board.png")
    print(ocr_txt)
