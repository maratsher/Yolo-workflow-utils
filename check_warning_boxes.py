"""
Скрипт проверки датасета на проблемные bounding boxes.
"""
import os
import tqdm

from argparse import ArgumentParser

from utils import collection_data


class BadListGenerator:
    def __init__(self, path_to_dataset, remove_boxes=False):
        self.path_to_dataset = path_to_dataset
        self.remove_boxes = remove_boxes
        self.filename_bad_list = os.path.join(os.getcwd(), 'bad_list_boxes')

    @staticmethod
    def is_bad_label(label):
        """ Функция проверяет условие плохого бокса

        :param label: list со строками [x, y, w, h].
        :return: bool.
        """
        label = [float(x) for x in label]
        if label[0] > 1 or label[0] <= 0 or label[1] > 1 or \
                label[1] <= 0 or label[2] > 1 or label[2] <= 0 or label[3] > 1 \
                or label[3] <= 0:
            return True
        else:
            return False

    @staticmethod
    def remove_image_and_label(label: str, image: str):
        label = '{}.txt'.format(label)
        try:
            os.remove(label)
        except FileNotFoundError:
            pass

        try:
            os.remove(image)
        except FileNotFoundError:
            pass

    def generate_bad_list(self):
        """Главная функция класса.
        Генерирует файл со списком плохих боксов.
        Удаляет проблемные боксы если установлени флаг.
        Удаляет файл изображения и файл разметки, если файл разметки
        пуст после удаления проблемных боксов."""
        print("INFO: The script started it's work!")
        images, labels = collection_data(self.path_to_dataset)
        lines_for_bad_list = []
        for label, image in tqdm.tqdm(zip(labels, images), total=len(labels)):
            with open('{}.txt'.format(label), 'r') as f:
                lines = f.readlines()
                right_boxes = []
                for index, line in enumerate(lines):
                    line_list = line.split()[1:]
                    if self.is_bad_label(line_list):
                        line_to_bad_list = 'label: {}.txt img_format: {} |' \
                                           ' line = {}, x = {}, y = {},' \
                                           ' width = {}, height = {}\n'\
                            .format(label, os.path.splitext(image)[-1], index + 1,
                                    line_list[0], line_list[1], line_list[2],
                                    line_list[3]
                                    )
                        lines_for_bad_list.append(line_to_bad_list)
                    else:
                        right_boxes.append(line)
            if self.remove_boxes:
                if not right_boxes:
                    self.remove_image_and_label(label, image)
                    continue
                with open('{}.txt'.format(label), 'w') as f:
                    text = ''.join(right_boxes)
                    f.write(text)
        if not lines_for_bad_list:
            print('INFO: No warnings found.')
        else:
            with open(self.filename_bad_list, 'w') as f:
                text = ''.join(lines_for_bad_list)
                f.write(text)
            print('INFO: Bad list generate complete!')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-d", "--dir_to_dataset",
                        required=True, help="The path to the dataset")
    parser.add_argument("-r", "--remove_boxes",
                        required=False, help="Remove bad boxes",
                        default=False, action='store_true')

    args = vars(parser.parse_args())

    path_to_dataset = args['dir_to_dataset']
    remove_boxes = bool(args['remove_boxes'])

    bad_list_generator = BadListGenerator(path_to_dataset, remove_boxes)
    bad_list_generator.generate_bad_list()
