"""
Здесь находятся вспомогательные функции для работы скриптов.
"""
import os
import re
from functools import partial

from numpy import random


def return_path_to_file(string):
    """Функция возвращает путь до файла из строки"""
    return re.findall(r'(\/.*?\.[\w:]+)', string)[0]


def return_format_image(string):
    """Функция возвращает расширение изображения исходя из bad_list"""
    return re.findall(r'img_format: (.[a-z]+)', string)[0]


def extension_filter(extension, filename):
    """Фильтр поиска текстовых файлов.

    :param filename: имя файла.
    :param extension: тип файла.

    :return: название файла.
    """
    format_file = {'image': ['.jpg', '.png', '.bmp', '.jpeg', '.gif'],
                   'video': ['.mp4'],
                   'text': ['.txt', '.xml']}

    return os.path.splitext(filename)[-1] in format_file[extension]


def collection_data(path_to_dataset, randomly=False):
    """Функция собирает данные названия файлов для последующей обработки.

    :param path_to_dataset: путь до датасета.

    :return: множество названий файлов изображений,
    множество названий текстовых файлов разметки.
    """
    list_file = []
    for dirs, paths, files in os.walk(path_to_dataset):
        for file in files:
            list_file.append(os.path.join(dirs, file))
    if randomly:
        list_file = random.permutation(list_file)
    list_file_image = set(filter(partial(extension_filter, 'image'), list_file))
    list_file_text = set(os.path.splitext(filename)[0] for filename in filter(
        partial(extension_filter, 'text'), list_file))
    return list_file_image, list_file_text


def collection_data_classifier(path_to_dataset):
    """Функция собирает данные названия файлов для последующей обработки.

    :param path_to_dataset: путь до датасета.

    :return: множество названий файлов изображений,
    множество названий текстовых файлов разметки.
    """
    dict_files = {}
    for paths, dirs, files in os.walk(path_to_dataset):
        for dir in dirs:
            dict_files[dir] = os.listdir(os.path.join(path_to_dataset, dir))
    return dict_files


def percentage_ratio(percent):
    """Функция получает количество аргументов для:
    обучающей, тестовой (валидационной) выборки.

    :param percent: проценты в виде '60_40'
    :return: аргументы для выборок.
    """
    train = percent.split('_')[0]
    test = percent.split('_')[1]
    if len(percent.split('_')) == 3:
        validate = percent.split('_')[2]
        return train, test, validate
    return train, test


def write_txt_file(list_images_csv, save_path, dir_sample):
    """Функция формирует csv файл с путями до картинок.

    :param list_images_csv: лист с изображениями.
    :param save_path: путь до папки с выборками.
    :param dir_sample: папка сохранения (train, test, validate).
    """
    with open('{}/{}.txt'.format(save_path, dir_sample), 'w') as file:
        file.writelines(list_images_csv)


def convert_labels(img, boxes_yolo, class_in_box=False, edited_box=False):
    """Функция трансформирует координаты bbox
     из стандартного формата в yolo-формат.

    :param img: изображение (массив).
    :param boxes_yolo: yolo boxes.
    :param: class_in_box: флаг добавления id класса в bbox.
    """
    def sorting(l1, l2):
        """Функция сортировки координат по убыванию.

        :param l1: координата 1.
        :param l2: координата 2.
        """
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    height, width, _ = img.shape
    yolo_bboxes_convert = []
    if edited_box:
        boxes_yolo = boxes_yolo.bounding_boxes
    for bbox in boxes_yolo:
        if edited_box:
            xmax, xmin = sorting(bbox.x1, bbox.x2)
            ymax, ymin = sorting(bbox.y1, bbox.y2)
        else:
            xmax, xmin = sorting(bbox[1], bbox[3])
            ymax, ymin = sorting(bbox[2], bbox[4])
        dw = 1./width
        dh = 1./height
        x = (xmin + xmax)/2.0
        y = (ymin + ymax)/2.0
        w = xmax - xmin
        h = ymax - ymin
        x = x * dw if x * dw > 0 else 0
        w = w * dw if w * dw > 0 else 0
        y = y * dh if y * dh > 0 else 0
        h = h * dh if h * dh > 0 else 0

        if class_in_box:
            if edited_box:
                yolo_bboxes_convert.append((int(bbox.label), x, y, w, h))
            else:
                yolo_bboxes_convert.append((int(bbox[0]), x, y, w, h))
        else:
            yolo_bboxes_convert.append((x, y, w, h))

    return yolo_bboxes_convert


def transform_coordinates(path_to_dataset, image, file_img, add_class_in_box=False):
    """Функция трансформирует координаты bbox
     из формата yolo в стандартный.

    :param image: изображение.
    :param file_img: название изображения.
    :param add_class_in_box: флаг добавления id класса в bbox.
    :return: лист с координатами bbox.
    """
    height, width, _ = image.shape
    with open(os.path.join(path_to_dataset,
                           os.path.splitext(file_img)[0] + '.txt'), 'r') as f:
        darknet_lines = f.readlines()
    new_list_bboxes = []
    for darknet_line in darknet_lines:
        class_id, xc, yc, w, h = map(float, darknet_line.split(' '))
        xl = int((xc - w / 2) * width)
        yl = int((yc - h / 2) * height)
        xr = int((xc + w / 2) * width)
        yr = int((yc + h / 2) * height)
        if add_class_in_box:
            new_list_bboxes.append([int(class_id), xl, yl, xr, yr])
        else:
            new_list_bboxes.append([xl, yl, xr, yr])
    return new_list_bboxes
