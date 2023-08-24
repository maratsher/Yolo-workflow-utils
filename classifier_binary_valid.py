import os
import csv
from argparse import ArgumentParser

import cv2
import numpy as np

from darknet import DarknetClassifier


class PreprocessingImage:
    @staticmethod
    def get_pixel(image: np.ndarray, x: int, y: int, c: int):
        im_h, im_w, im_c = image.shape
        assert x < im_w and y < im_h and c < im_c
        return image[y][x][c]

    @staticmethod
    def set_pixel(image: np.ndarray, x: int, y: int, c: int, val: float):
        im_h, im_w, im_c = image.shape
        if x < 0 or y < 0 or c < 0 or x >= im_w or y >= im_h or c >= im_c:
            return
        assert x < im_w and y < im_h and c < im_c
        image[y][x][c] = val

    @staticmethod
    def add_pixel(image: np.ndarray, x: int, y: int, c: int, val: float):
        im_h, im_w, im_c = image.shape
        assert x < im_w and y < im_h and c < im_c
        image[y][x][c] += val

    @staticmethod
    def make_empty_image(w: int, h: int, c: int):
        out = np.zeros((h, w, c))
        return out

    def make_image(self, w: int, h: int, c: int):
        out = self.make_empty_image(w, h, c)
        return out

    @staticmethod
    def constrain_int(a, min_: int, max_: int):
        if a < min_:
            return int(min_)
        if a > max_:
            return int(max_)
        return int(a)

    def resize_image(self, image: np.ndarray, w: int, h: int):
        im_h, im_w, im_c = image.shape

        if im_w == w and im_h == h:
            return image

        resized = self.make_image(w, h, im_c)
        part = self.make_image(w, im_h, im_c)

        w_scale = float(im_w - 1) / (w - 1)
        h_scale = float(im_h - 1) / (h - 1)

        for k in range(0, 1000000):
            if k >= im_c:
                break
            for r in range(0, 1000000):
                if r >= im_h:
                    break
                for c in range(0, 1000000):
                    if c >= w:
                        break
                    val = float(0)
                    if c == w - 1 or im_w == 1:
                        try:
                            val = (self.get_pixel(image, im_w - 1, r, k)) / 255.0
                        except AssertionError:
                            break
                    else:
                        sx = float(c * w_scale)
                        ix = int(sx)
                        dx = float(sx - ix)
                        try:
                            val = ((1 - dx) *
                                   self.get_pixel(image, ix, r, k) + dx *
                                   self.get_pixel(image, ix + 1, r, k)) \
                                  / 255.0
                        except AssertionError:
                            break
                    try:
                        self.set_pixel(part, c, r, k, val)
                    except AssertionError:
                        break

        for k in range(0, 1000000):
            if k >= im_c:
                break
            for r in range(0, 1000000):
                if r >= h:
                    break
                sy = float(r * h_scale)
                iy = int(sy)
                dy = float(sy - iy)
                for c in range(0, 1000000):
                    if c >= w:
                        break
                    try:
                        val = (1 - dy) * self.get_pixel(part, c, iy, k)
                        self.set_pixel(resized, c, r, k, val)
                    except AssertionError:
                        break
                if r == h - 1 or im_h == 1:
                    continue

                for c in range(0, 1000000):
                    if c >= w:
                        break
                    try:
                        val = float(dy * self.get_pixel(part, c, iy + 1, k))
                        self.add_pixel(resized, c, r, k, val)
                    except AssertionError:
                        break

        return resized

    def resize_min(self, image: np.ndarray, min_: int):
        h, w, _ = image.shape
        im_h, im_w, _ = image.shape

        if w < h:
            h = int((h * min_) / w)
            w = min_
        else:
            w = (w * min_) / h
            h = min_
        if w == im_w and h == im_h:
            return image

        w = int(w)
        h = int(h)
        resized = self.resize_image(image, w, h)

        return resized

    def crop_image(self, image: np.ndarray, dx: int, dy: int, w: int, h: int):
        im_h, im_w, im_c = image.shape

        cropped = self.make_image(w, h, im_c)
        i, j, k = 0, 0, 0
        while k < im_c:
            j = 0
            while j < h:
                i = 0
                while i < w:
                    r = j + dy
                    c = i + dx
                    val = float(0)
                    r = self.constrain_int(r, 0, im_h - 1)
                    c = self.constrain_int(c, 0, im_w - 1)
                    if r >= 0 and r < im_h and c >= 0 and c < im_w:
                        val = self.get_pixel(image, c, r, k)
                    self.set_pixel(cropped, i, j, k, val)
                    i += 1
                j += 1
            k += 1
        return cropped


def main():
    classifier = DarknetClassifier(
        config_path=config_path,
        weights_path=save_path,
        meta_path=meta_path
    )

    tp, tn, fp, fn = 0, 0, 0, 0

    data_list = []
    for i, filename in enumerate(images):
        image = cv2.imread(filename.rstrip('\n'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if with_resize_cv2:
            image_preprocessing = cv2.resize(
                image, (classifier.net_w, classifier.net_h),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            # implementation of the original pre-processing from the darknet.
            preprocessing_image = PreprocessingImage()
            resized = preprocessing_image.resize_min(image, classifier.net_w)
            im_h, im_w, im_c = resized.shape
            image_preprocessing = preprocessing_image.crop_image(
                resized,
                (im_w - classifier.net_w) / 2,
                (im_h - classifier.net_h) / 2,
                classifier.net_w,
                classifier.net_h
            )

        predict = classifier.classify_image(image_preprocessing)
        probability = predict[0][1]

        print(f'{i}: {probability}')

        right_answer = os.path.splitext(filename)[0].split('_')[2]
        current_answer = predict[0][0].decode("utf-8")

        if right_answer == classifier.names[1] and current_answer == classifier.names[1]:
            tp += 1
        elif right_answer == classifier.names[0] and current_answer == classifier.names[0]:
            tn += 1
        elif right_answer == classifier.names[1] and current_answer == classifier.names[0]:
            fn += 1
            data_list.append([filename, current_answer, right_answer, probability])
        elif right_answer == classifier.names[0] and current_answer == classifier.names[1]:
            fp += 1
            data_list.append([filename, current_answer, right_answer, probability])

    if create_doc_analyze:
        with open('error_analyze.csv', 'w') as f_:
            writer = csv.writer(f_)
            writer.writerow(['filename', 'curr_class', 'right_class', 'probability'])
            for data in data_list:
                writer.writerow(data)

    print(f'TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}')

    accuracy = round((tp + tn) / (tp + tn + fp + fn), 6)

    try:
        precision = round(tp / (tp + fp), 2)
    except ZeroDivisionError:
        precision = 'NaN'
    try:
        recall = round(tp / (tp + fn), 2)
    except ZeroDivisionError:
        recall = 'NaN'
    if precision != 'NaN' and recall != 'NaN':
        f_measure = round((2 * precision * recall) / (precision + recall), 2)
    else:
        f_measure = 'NaN'

    print(f'Acc: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f_measure}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", required=True)
    parser.add_argument("-w", "--weights_path", required=True)
    parser.add_argument("-m", "--meta_path", required=True)

    parser.add_argument("-r", "--resize_cv2", required=False, default=True, action='store_true')

    parser.add_argument('-cd', "--create_doc", required=False, default=False, action='store_true')

    parser.add_argument("-d", "--path_dataset", required=False)

    args = vars(parser.parse_args())

    config_path = args['config_path']
    save_path = args['weights_path']
    meta_path = args['meta_path']

    with_resize_cv2 = args['resize_cv2']

    create_doc_analyze = args['create_doc']

    path_to_data_txt = args['path_dataset']
    with open(path_to_data_txt, 'r') as f:
        images = f.readlines()

    main()
