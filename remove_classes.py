"""
Скрипт удаления классов из файлов разметки.
"""
import os
import shutil
from argparse import ArgumentParser

from tqdm import tqdm

from utils import collection_data


def count_classes_in_file():
    """Функция подсчета количества классов в файле classes"""
    with open(path_to_classes, 'r') as f:
        file_classes = f.readlines()
        return len(file_classes) - 1


def create_markup_list(file_txt):
    """Функция создает новый список для файла разметки, если
    номер класса объекта не выходит за границы количества классов.
    """
    new_list_labels = []
    with open(os.path.join(path_to_dataset, file_txt), 'r') as f:
        original_list_labels = f.readlines()
        for line in original_list_labels:
            try:
                if int(line.split(' ')[0]) <= count_classes:
                    new_list_labels.append(line)
            except ValueError:
                pass
    return new_list_labels, original_list_labels


def clear_not_needed_classes_in_markup():
    """Функция выполняет одно из действий.
    Если файл разметки пустой, то выполняется перемещение изображения и файла разметки
    в папку negative.

    Если новый лист разметки не совпадает с оригинальным, то производится
    замена файла разметки без учета классов, которые не подходят под параметр
    количества классов.
    """
    print("INFO: The script started it's work!")
    list_file_image, list_file_text = collection_data(path_to_dataset)
    os.makedirs(os.path.join(path_to_dataset, 'negative'), exist_ok=True)
    for filename in tqdm(list_file_image):
        file_txt = '{}.txt'.format(os.path.basename(filename).split('.')[0])
        list_labels, original_list_labels = create_markup_list(file_txt)
        if list_labels != original_list_labels:
            with open(os.path.join(path_to_dataset, file_txt), 'w') as f:
                f.writelines(list_labels)
        if len(list_labels) == 0:
            shutil.copy(filename,
                        os.path.join(path_to_dataset, 'negative', os.path.basename(filename)))

            shutil.copy(os.path.join(path_to_dataset, file_txt),
                        os.path.join(path_to_dataset, 'negative', file_txt))

    if not os.listdir(os.path.join(path_to_dataset, 'negative')):
        shutil.rmtree(os.path.join(path_to_dataset, 'negative'))
        print('INFO: No warnings found.')
    else:
        print('Created a folder with negatives: {}'.format(os.path.join(path_to_dataset, 'negative')))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir_to_dataset", required=False,
                        help="The path to the dataset")
    parser.add_argument("-c", "--classes", required=False,
                        help="The path to file classes")

    args = vars(parser.parse_args())

    path_to_dataset = args['dir_to_dataset']
    path_to_classes = args['classes']

    count_classes = count_classes_in_file()
    clear_not_needed_classes_in_markup()
