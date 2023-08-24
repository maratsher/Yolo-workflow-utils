"""
Скрипт тестирования yolo-моделей.
"""
import os
import csv
import time
from copy import deepcopy
from datetime import timedelta

import cv2
import numpy as np
import imgaug as ia
from tqdm import tqdm
from argparse import ArgumentParser

from darknet import YoloV3, net_size_image, net_classes_names
from utils import collection_data, transform_coordinates

METRICS = {}


def transform_coordinates_letter_box(image, boxes):
    """Функция переводит координаты yolo-формата в стандартный формат.

    :param image: изображение в array.
    :return: лист с преобразованными координатами bboxes.
    """
    width, height, _ = image.shape
    new_list_bboxes = []

    for bbox in boxes:
        ia.seed(1)
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=bbox[1], x2=bbox[3], y1=bbox[2], y2=bbox[4],
                           label=bbox[0])],
            shape=(input_height_img, input_width_img, 3))
        image_rescaled = ia.imresize_single_image(image, (width, height))
        bbs_rescaled = bbs.on(image_rescaled)
        new_list_bboxes.append([bbs_rescaled.bounding_boxes[0].label,
                                bbs_rescaled.bounding_boxes[0].x1,
                                bbs_rescaled.bounding_boxes[0].y1,
                                bbs_rescaled.bounding_boxes[0].x2,
                                bbs_rescaled.bounding_boxes[0].y2])

    return new_list_bboxes


def draw_bb(image, file_img, boxes_ground_true, boxes_yolo, i):
    """Функция отрисовки bboxes на изображении.

    :param image: изображение (массив).
    :param file_img: путь до изображения.
    :param boxes_ground_true: ground true разметка.
    :param boxes_yolo: yolo разметка.
    :param i: номер изображения.
    """
    # ToDO: Шапку с количеством боксов пропорциональнее.
    if int(show_img) > i and (boxes_yolo or boxes_ground_true) or save_image:
        colors = [(238, 104, 123), (0, 0, 255), (0, 255, 0), (255, 0, 0),
                  (14, 195, 223), (24, 189, 208), (255, 0, 255), (51, 0, 102),
                  (0, 140, 255), (170, 232, 238), (0, 100, 0), (49, 79, 47),
                  (130, 0, 75), (203, 192, 255), (19, 69, 139), (102, 105, 105)]
        font_scale = 0.7
        if show_ground_true_boxes:
            for bbox_gt in boxes_ground_true:
                try:
                    color_box = colors[bbox_gt[0]]
                except IndexError:
                    color_box = colors[-1]
                cv2.rectangle(image,
                              (int(bbox_gt[1]), int(bbox_gt[2])),
                              (int(bbox_gt[3]), int(bbox_gt[4])), color_box, 2)
                cv2.rectangle(image,
                              (int(bbox_gt[1] + 1), int(bbox_gt[2] + 1)),
                              (int(bbox_gt[3] + 1), int(bbox_gt[4] + 1)), (0, 0, 0), 1)
                text_size, _ = cv2.getTextSize(names[bbox_gt[0]],
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale,
                                               cv2.LINE_4)
                cv2.rectangle(image,
                              (int(bbox_gt[1]), int(bbox_gt[2] - 25)),
                              (int(bbox_gt[1] + text_size[0]), int(bbox_gt[2])), color_box, thickness=cv2.FILLED)
                cv2.rectangle(image,
                              (int(bbox_gt[1] + 1), int(bbox_gt[2] - 24)),
                              (int(bbox_gt[1] + text_size[0]), int(bbox_gt[2])),
                              (0, 0, 0))
                cv2.putText(image,
                            text=names[bbox_gt[0]],
                            org=(int(bbox_gt[1]), int(bbox_gt[2] - 7)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale,
                            color=(255, 255, 255),
                            lineType=cv2.LINE_4)
            cv2.rectangle(image, (0, 38), (350, 76), (0, 0, 0), -1)
            cv2.putText(image,
                        text='orig bbox: {}'.format(len(boxes_ground_true)),
                        org=(5, 60),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        lineType=cv2.LINE_4)
        for bbox_y in boxes_yolo:
            try:
                color_box = colors[bbox_y[0]]
            except IndexError:
                color_box = colors[-1]
            cv2.rectangle(image,
                          (int(bbox_y[1]), int(bbox_y[2])),
                          (int(bbox_y[3]), int(bbox_y[4])), color_box, 2)
            cv2.rectangle(image,
                          (int(bbox_y[1] + 1), int(bbox_y[2] + 1)),
                          (int(bbox_y[3] + 1), int(bbox_y[4] + 1)), (255, 255, 255), 1)
            text_size, _ = cv2.getTextSize(names[bbox_y[0]],
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale,
                                           cv2.LINE_4)
            cv2.rectangle(image,
                          (int(bbox_y[1]), int(bbox_y[4])),
                          (int(bbox_y[1] + text_size[0]), int(bbox_y[4] + 25)), color_box, thickness=cv2.FILLED)
            cv2.rectangle(image,
                          (int(bbox_y[1] + 1), int(bbox_y[4])),
                          (int(bbox_y[1] + text_size[0]), int(bbox_y[4] + 26)),
                          (255, 255, 255))
            cv2.putText(image,
                        text=names[bbox_y[0]],
                        org=(int(bbox_y[1]), int(bbox_y[4] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=(255, 255, 255),
                        lineType=cv2.LINE_4)
        if show_ground_true_boxes:
            cv2.rectangle(image, (0, 0), (350, 38), (0, 0, 0), -1)
            cv2.putText(image,
                        text='yolo bbox: {}'.format(len(boxes_yolo)),
                        org=(5, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 0, 170),
                        lineType=cv2.LINE_AA)
            cv2.rectangle(image, (0, 76), (350, 114), (0, 0, 0), -1)
            cv2.putText(image,
                        text=str(os.path.split(file_img)[-1]),
                        org=(5, 95),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        lineType=cv2.LINE_AA)
    if save_image:
        os.makedirs(os.path.join(os.getcwd(), 'bbox_images'), exist_ok=True)
        cv2.imwrite(os.path.join(os.getcwd(), 'bbox_images', os.path.basename(file_img)), image)
    if int(show_img) > i and (boxes_yolo or boxes_ground_true):
        cv2.imshow('image', image)
        cv2.waitKey(0)
    else:
        cv2.destroyAllWindows()


def calculation_iou(file_img, boxes_ground_true, boxes_yolo):
    """Функция подсчитывает iou в соответствии с оригинальной разметкой и
    разметкой тестируемой модели.

    :param file_img: путь до изображения.
    :param boxes_ground_true: оригинальная разметка.
    :param boxes_yolo: разметка тестируемой модели.
    """
    # ToDO: Проверить правильность рассчета class_obj.
    iou, avg_score, worst_iou, status = 0, 0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    class_obj = False
    if boxes_ground_true and boxes_yolo:
        same_boxes_counter = 0
        list_boxes_yolo = deepcopy(boxes_yolo)
        iou_list = []
        for boxGT in boxes_ground_true:
            max_iou = 0
            class_gt = boxGT[0]
            same_box_from_yolo = None
            for boxY in list_boxes_yolo:
                class_yolo = boxY[0]
                x1 = max(boxGT[1], boxY[1])
                y1 = max(boxGT[2], boxY[2])
                x2 = min(boxGT[3], boxY[3])
                y2 = min(boxGT[4], boxY[4])
                interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
                boxAArea = (boxGT[3] - boxGT[1] + 1) * (boxGT[4] - boxGT[2] + 1)
                boxBArea = (boxY[3] - boxY[1] + 1) * (boxY[4] - boxY[2] + 1)
                iou = interArea / float(boxAArea + boxBArea - interArea)
                if iou > max_iou:
                    max_iou = iou
                    class_obj = class_gt == class_yolo
                    same_box_from_yolo = boxY
            if max_iou >= threshold and class_obj:
                same_boxes_counter = same_boxes_counter + 1
                list_boxes_yolo.remove(same_box_from_yolo)
                iou_list.append(max_iou)
        iou_list = sorted(iou_list, reverse=True)
        TP = len(iou_list)
        FP = len(list_boxes_yolo)
        FN = len(boxes_ground_true) - same_boxes_counter
        if iou_list:
            iou = str(iou_list)[1:-1]
            avg_score = np.mean(iou_list)
            worst_iou = iou_list[0]
        else:
            iou = 0.0
            avg_score = 0
            worst_iou = 0
        status = 1 if TP == len(boxes_ground_true) else 0

    elif not boxes_ground_true and not boxes_yolo:
        status = 1
        iou = ''
        avg_score = 1
        worst_iou = ''
    elif len(boxes_ground_true) == 0 and boxes_yolo:
        FP = len(boxes_yolo)
        iou = 0.0
        avg_score = 0
        worst_iou = 0
    else:
        FN = len(boxes_ground_true)
        iou = 0.0
        avg_score = 0
        worst_iou = 0

    METRICS[file_img] = {'iou': iou,
                         'avg_score': avg_score,
                         'worst_iou': worst_iou,
                         'TP': TP,
                         'TN': TN,
                         'FP': FP,
                         'FN': FN,
                         'status': status}


def calculation_basic_metrics():
    """Функция формирует txt-файл с основными метриками тестируемой модели.
    Производится подсчет TP, TN, FP, FN.
    Accuracy, precision, recall, f1-score, avg_score.
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    average_score = []
    for k, v in METRICS.items():
        TP = TP + v['TP']
        FN = FN + v['FN']
        FP = FP + v['FP']
        TN = TN + v['TN']
        if isinstance(v['avg_score'], float or int):
            average_score.append(v['avg_score'])

    accuracy = ((TP + TN) / (TP + TN + FP + FN))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    if (TP + FP) > 0:
        average_score = (sum(average_score) / (TP + FP)) * 100
    else:
        average_score = 0
    basic_metrics = 'INFO METRICS:\n' \
                    '===========================\n' \
                    'all images: {}\n' \
                    'TP: {} | TN: {} | FP: {} | FN: {}\n' \
                    'accuracy: {}\n' \
                    'precision: {}\n' \
                    'recall: {}\n' \
                    'f1-score: {}\n' \
                    'average_score: {}\n'.format(len(METRICS), TP, TN, FP, FN,
                                               round(accuracy, 2),
                                               round(precision, 2),
                                               round(recall, 2),
                                               round(f1_score, 2),
                                               average_score)

    with open('metrics.txt', 'w') as f:
        f.writelines(basic_metrics)

    print(basic_metrics)
    print('Metrics txt formed: {}'.format(
        os.path.join(os.getcwd(), 'metrics.txt')))
    if save_image:
        print('INFO: Images saved in: {}'.format(os.path.join(os.getcwd(), 'bbox_images')))


def create_csv_file():
    """Функция формирует csv-файл с данными об изображении:
        file, iou, avg_score, worst_iou, TP, FN, FP, status.
    """
    with open('statistics_test.csv', 'w') as f:
        fieldnames = ['file', 'iou', 'avg_score', 'worst_iou',
                      'TP', 'TN', 'FP', 'FN', 'status']
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for k, v in METRICS.items():
            wr.writerow({'file': k,
                         'iou': v['iou'],
                         'avg_score': v['avg_score'],
                         'worst_iou': v['worst_iou'],
                         'TP': v['TP'],
                         'TN': v['TN'],
                         'FP': v['FP'],
                         'FN': v['FN'],
                         'status': v['status']})

    print('Statistics test csv formed: {}\n'.format(
        os.path.join(os.getcwd(), 'statistics_test.csv')))


def model_testing():
    """Основная функция тестирования yolo-моделей.
    """
    start_time = time.time()
    list_file_image, list_file_text = collection_data(path_to_dataset)
    i = 0
    for file_img in tqdm(list(list_file_image)):
        raw_image = cv2.imread(file_img)
        boxes_ground_true = transform_coordinates(path_to_dataset,
                                                  raw_image,
                                                  file_img,
                                                  add_class_in_box=True)
        boxes_yolo = yolo.get_boxes(raw_image,
                                    thresh=threshold,
                                    hier_thresh=hier_thresh,
                                    nms=nms,
                                    letter_box=letter_box,
                                    classes=classes)
        if boxes_yolo and not letter_box:
            boxes_yolo = transform_coordinates_letter_box(raw_image, boxes_yolo)
        draw_bb(raw_image, file_img, boxes_ground_true, boxes_yolo, i)
        i = i + 1
        calculation_iou(file_img, boxes_ground_true, boxes_yolo)

    create_csv_file()
    calculation_basic_metrics()
    finish_time = timedelta(seconds=(time.time() - start_time))
    print('===========================')
    print('INFO: The run time of the script took {}.'.format(finish_time))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir_to_dataset",
                        required=False, help="The path to the dataset")
    parser.add_argument("-cl", "--classes",
                        required=False, help="The path to the file classes")
    parser.add_argument("-cfg", "--cfg_path",
                        required=False, help="Path to yolo config")
    parser.add_argument("-dp", "--data_path",
                        required=False, help="Path to data")
    parser.add_argument("-w", "--weights_path",
                        required=False, help="Path to weights")
    parser.add_argument("-l", "--letter_box",
                        required=False, help="letter_box",
                        default=False, action='store_true')
    parser.add_argument("-t", "--threshold",
                        required=False, help="Threshold")
    parser.add_argument("-ht", "--hier_thresh",
                        required=False, help="Hier threshold")
    parser.add_argument("-n", "--nms",
                        required=False, help="nms")
    parser.add_argument("-sh", "--show_image",
                        required=False, help="show_img")
    parser.add_argument("-gt", "--show_ground_true_boxes",
                        required=False, help="show_ground_true_boxes",
                        default=False, action='store_true')
    parser.add_argument("-si", "--save_image",
                        required=False, help="show_img",
                        default=False, action='store_true')

    args = vars(parser.parse_args())

    path_to_dataset = args['dir_to_dataset']
    cfg_path = args['cfg_path']
    data_path = args['data_path']
    weights_path = args['weights_path']
    letter_box = bool(args['letter_box'])
    show_img = args['show_image'] if args['show_image'] else 0
    show_ground_true_boxes = bool(args['show_ground_true_boxes'])
    save_image = bool(args['save_image'])

    threshold = float(args['threshold']) if args['threshold'] else 0.5
    hier_thresh = float(args['hier_thresh']) if args['hier_thresh'] else 0.5
    nms = float(args['nms']) if args['nms'] else 0.5

    yolo = YoloV3(cfg_path, weights_path, data_path)
    input_width_img, input_height_img = net_size_image(cfg_path, weights_path)

    classes, names = net_classes_names(data_path)

    model_testing()
