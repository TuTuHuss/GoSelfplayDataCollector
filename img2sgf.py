import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from paddleocr import PaddleOCR


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
            cv2.circle(stone, (y_cent, x_cent), radius, (0, 0, 255), 3)

            x_left, x_right = max(0, x_cent - radius), min(stone.shape[0], x_cent + radius)
            y_left, y_right = max(0, y_cent - radius), min(stone.shape[1], y_cent + radius)
            patches[-1].append(checkerboard[x_left: x_right, y_left: y_right, :])

    for cir in circles:
        cv2.circle(stone, (int(cir[0]), int(cir[1])), int(cir[2]), (0, 255, 0), 3)
    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(stone)
    plt.show()

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
            # pat = cv2.GaussianBlur(pat, (3, 3), 1)
            # pat_x = cv2.Sobel(pat, -1, 1, 0)
            # pat_y = cv2.Sobel(pat, -1, 0, 1)
            # scale_x = cv2.convertScaleAbs(pat_x)
            # scale_y = cv2.convertScaleAbs(pat_y)
            # pat = cv2.addWeighted(scale_x, 0.5, scale_y, 0.5, 0)

            pat = pat[int(pat.shape[0] * margin_ratio): int(pat.shape[0] * (1 - margin_ratio)),
                  int(pat.shape[1] * margin_ratio): int(pat.shape[1] * (1 - margin_ratio)), :]

            # pat = cv2.morphologyEx(pat, cv2.MORPH_CLOSE, np.ones((3, 3)))

            # if np.mean(pat) > 115:
            #     pat = 255 - pat

            padded_img = np.pad(pat,
                                ((before_x, PADDED_IMG_SIZE - before_x - pat.shape[0]),
                                 (before_y, PADDED_IMG_SIZE - before_y - pat.shape[1]),
                                 (0, 0)))

            res = ocr.ocr(img=padded_img, det=True)[0]
            if res is None:
                ret[-1].append(None)
            else:
                ret[-1].append(res[0][1][0])
            # plt.figure(figsize=(10, 10), dpi=80)
            # cv2.putText(img=padded_img, text=str(ret[-1][-1]), org=(50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            #             fontScale=0.5,
            #             color=(0, 255, 0), thickness=1)
            # plt.imshow(padded_img)
            # plt.show()
    return ret


def post_process(raw_ocr):
    for j in range(len(raw_ocr)):
        for i in range(len(raw_ocr[j])):
            if isinstance(raw_ocr[j][i], str):
                raw_ocr[j][i] = raw_ocr[j][i].upper()
                func = lambda c: c in "0123456789ABCDEFG△□"
                raw_ocr[j][i] = ''.join(filter(func, raw_ocr[j][i]))
                if len(raw_ocr[j][i]) == 0:
                    raw_ocr[j][i] = None


def get_white_ratio(pat):
    margin_ratio = 0.25
    pat = pat[int(pat.shape[0] * margin_ratio): int(pat.shape[0] * (1 - margin_ratio)),
          int(pat.shape[1] * margin_ratio): int(pat.shape[1] * (1 - margin_ratio)), :]

    return np.sum(pat) / np.sum(255 * np.ones_like(pat))


def get_prediction(ocr_info, patches):
    res = copy.deepcopy(ocr_info)
    for i in range(len(ocr_info)):
        cols = len(ocr_info[i])
        for j in range(cols):
            x_ocr = ocr_info[i][j]
            x_pat = patches[i][j]

            if x_ocr is not None and x_ocr in 'ABC':
                res[i][j] = f'.({x_ocr})'
            elif x_ocr is None:
                white_ratio = get_white_ratio(x_pat)
                if white_ratio > 0.8:
                    res[i][j] = f'o'
                elif white_ratio < 0.2:
                    res[i][j] = f'#'
                else:
                    res[i][j] = f'.'
            else:
                white_ratio = get_white_ratio(x_pat)
                if white_ratio > 1 - white_ratio:
                    res[i][j] = f'o({x_ocr})'
                else:
                    res[i][j] = f'#({x_ocr})'
    return res


if __name__ == '__main__':
    image = load_img("./data/6_board.png")
    segmented_patches = patchify(image)

    print(f'Total number of patches: {sum([len(ll) for ll in segmented_patches])}')

    # OCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", rec_model_dir='asdf')
    ocr_res = batch_ocr(segmented_patches)
    post_process(ocr_res)

    final_res = get_prediction(ocr_res, segmented_patches)

    for l in final_res:
        print(l)
