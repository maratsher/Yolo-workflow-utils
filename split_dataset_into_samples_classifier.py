"""
Скрипт формирования выборок для классификатора
"""
import os
import time
import random
from numpy import random as np_random
import shutil
from datetime import timedelta
from multiprocessing import Pool
from argparse import ArgumentParser

from utils import collection_data_classifier, percentage_ratio, write_txt_file


def get_all_count_files(path_to_dataset):
    """Получает количество всех файлов в директории датасета.

    :param path_to_dataset: путь до датасета.
    :return: количетсво всех файлов.
    """
    counter_dirs_files = {}
    if len(next(os.walk(path_to_dataset))[1]) > 0:
        for path, dirs, files in os.walk(path_to_dataset):
            for dir in dirs:
                counter_dirs_files[dir] = len(os.listdir(os.path.join(path, dir)))
        return counter_dirs_files


def count_number_sample_files(count_all_files, sampling):
    """Функция возвращает количество файлов для каждой выборки,
    основываясь на процентном соотношении.

    :param count_all_files: количество всех файлов датасета.
    :param sampling: процентное соотношение выборки.
    :return: количество файлов для тренеровочного, тестового,
     (валидационного) набора данных
    """
    for k, v in count_all_files.items():
        train = round((count_all_files[k] * int(sampling[0])) / 100)
        test = round((count_all_files[k] * int(sampling[1])) / 100)
        if len(sampling) == 3:
            validate = round((count_all_files[k] * int(sampling[2])) / 100)
            count_all_files[k] = (train, test, validate)
        else:
            count_all_files[k] = (train, test)
    return count_all_files


def create_lists_sampling(samples, sampling):
    """Функция формирует списки файлов для последующего формирования выборок.

    :param samples: количество файлов для выборки.
    :return: лист с файлами разметки, лист с обучающими изображениями,
    лист с тестовыми изображениями, (лист с валидационными изображениями).
    """
    dict_files = collection_data_classifier(path_to_dataset)
    train_images = {}
    test_images = {}
    validate_images = {}
    if not dict_files:
        print('INFO: There is no data to process in the folder.')
        return
    if len(sampling) == 2:
        for k, v in dict_files.items():
            train_images[k] = v[0:samples[k][0]]
            test_images[k] = v[samples[k][0]:samples[k][0] + samples[k][1]]
        return train_images, test_images
    else:
        for k, v in dict_files.items():
            train_images[k] = v[0:samples[k][0]]
            test_images[k] = v[samples[k][0]:samples[k][0] + samples[k][1]]
            validate_images[k] = v[samples[k][0] + samples[k][1]:
                                   samples[k][0] + samples[k][1] + samples[k][2]]
        return train_images, test_images, validate_images


def new_name_file(classes, extension_file):
    """Функция создает новое названия изображения.

    :param classes: класс изображения.
    :param extension_file: расширение изображения.
    :return: новое название изображения.
    """
    random_number_1 = random.randint(0, 1000000)
    random_number_2 = random.randint(0, 1000)
    random_number = random_number_1 + random_number_2
    new_filename = '{}_{}{}'.format(random_number, classes, extension_file)

    return new_filename


def form_samples(dict_files, dir_sample):
    """Функция формирования выборок.

    :param dict_files: лист с изображениями.
    :param dir_sample: папка сохранения (train, test, validate).
    """
    list_images_txt = []
    os.makedirs(os.path.join(save_path, dir_sample), exist_ok=True)
    for classes, files in dict_files.items():
        for file in files:
            current_filename = os.path.splitext(file)[0]
            extension_file = os.path.splitext(file)[-1]
            new_filename = new_name_file(classes, extension_file)
            check_isfile = os.path.isfile(os.path.join(save_path, dir_sample, new_filename))
            if check_isfile:
                while check_isfile:
                    new_filename = new_name_file(classes, extension_file)
                    check_isfile = os.path.isfile(os.path.join(save_path, dir_sample, new_filename))
            shutil.move(os.path.join(path_to_dataset, classes, '{}{}'.format(current_filename, extension_file)),
                        os.path.join(save_path, dir_sample, new_filename))
            list_images_txt.append('{}\n'.format(
                os.path.join(save_path, dir_sample, new_filename)))
    if randomly:
        list_images_txt = np_random.permutation(list_images_txt)
    write_txt_file(list_images_txt, save_path, dir_sample)


def create_classes_list(count_all_files):
    """Функция формирует classes.txt файл со всеми классами.

    :param count_all_files: словарь с ключами=классами.
    """
    classes = list(count_all_files.keys())
    classes = [class_ + '\n' for class_ in classes]
    with open(os.path.join(save_path, 'classes.txt'), 'w') as f:
        text = ''.join(classes)
        f.writelines(text)


def main(path_to_dataset):
    """Функция собирает данные названия файлов для последующей обработки.

    :param path_to_dataset: путь до датасета.

    :return: множество названий файлов изображений,
    множество названий текстовых файлов разметки.
    """
    start_time = time.time()
    sampling = percentage_ratio(ratio)
    count_all_files = get_all_count_files(path_to_dataset)
    samples = count_number_sample_files(count_all_files, sampling)
    samplings = create_lists_sampling(samples, sampling)
    if not samplings:
        return
    print("INFO: The script started it's work!")
    if len(sampling) == 2:
        p = Pool(processes=3)
        p.apply_async(form_samples, args=(samplings[0], 'train'))
        p.apply_async(form_samples, args=(samplings[1], 'test'))
        p.close()
        p.join()
    else:
        p = Pool(processes=3)
        p.apply_async(form_samples, args=(samplings[0], 'train'))
        p.apply_async(form_samples, args=(samplings[1], 'test'))
        p.apply_async(form_samples, args=(samplings[2], 'validate'))
        p.close()
        p.join()
    create_classes_list(count_all_files)
    finish_time = timedelta(seconds=(time.time() - start_time))
    print('===========================')
    print('INFO: The run time of the script took {}.'.format(finish_time))
    print('INFO: File generated {0}.txt - {1}/{0}.txt'.format('train', save_path))
    print('INFO: File generated {0}.txt - {1}/{0}.txt'.format('test', save_path))
    if len(sampling) == 3:
        print('INFO: File generated {0}.txt - {1}/{0}.txt'.format('validate', save_path))
    print('INFO: No warnings found.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir_to_dataset", required=False, help="The path to the dataset")
    parser.add_argument("-s", "--save_path", required=False, help="The path to the save dataset")
    parser.add_argument("-sr", "--sample_ratio", required=False, help="Sample ratio. "
                                                                      "Indicate in the following order:"
                                                                      " train, test, validation (if required)")
    parser.add_argument("-r", "--random", required=False, help="Random data copying", action='store_true')

    args = vars(parser.parse_args())
    path_to_dataset = args['dir_to_dataset']
    save_path = args['save_path']
    ratio = args['sample_ratio']
    randomly = args['random']

    main(path_to_dataset)
