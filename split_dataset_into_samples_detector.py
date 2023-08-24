"""
Скрипт формирования выборок для детектора
"""
import os
import time
import shutil
from datetime import timedelta
from multiprocessing import Pool
from argparse import ArgumentParser

from utils import collection_data, percentage_ratio, write_txt_file


def get_all_count_files(path_to_dataset):
    """Получает количество всех файлов в директории датасета.

    :param path_to_dataset: путь до датасета.
    :return: количетсво всех файлов.
    """
    if len(next(os.walk(path_to_dataset))[1]) > 0:
        file = 0
        for path, dirs, files in os.walk(path_to_dataset):
            file += len(files)
        all_files = round((file - len(dirs)) / 2)
        return all_files
    else:
        all_files = round(len(next(os.walk(path_to_dataset))[2]) / 2)
        return all_files


def count_number_sample_files(count_all_files, sampling):
    """Функция возвращает количество файлов для каждой выборки,
    основываясь на процентном соотношении.

    :param count_all_files: количество всех файлов датасета.
    :param sampling: процентное соотношение выборки.
    :return: количество файлов для тренеровочного, тестового,
     (валидационного) набора данных
    """
    train = round((count_all_files * int(sampling[0])) / 100)
    test = round((count_all_files * int(sampling[1])) / 100)
    if len(sampling) == 3:
        validate = round((count_all_files * int(sampling[2])) / 100)
        return train, test, validate
    return train, test


def create_lists_sampling(samples):
    """Функция формирует списки файлов для последующего формирования выборок.

    :param samples: количество файлов для выборки.
    :return: лист с файлами разметки, лист с обучающими изображениями,
    лист с тестовыми изображениями, (лист с валидационными изображениями).
    """
    list_file_image, list_file_text = collection_data(path_to_dataset, randomly=randomly)
    if len(list_file_text) == 0:
        print('INFO: Missing markup files for images.')
        return
    if not list_file_image:
        print('INFO: There is no data to process in the folder.')
        return
    if len(samples) == 2:
        train_images = set(list(list_file_image)[0:samples[0]])
        test_images = set(list(list_file_image)[samples[0]:samples[0] + samples[1]])
        return train_images, test_images
    else:
        train_images = set(list(list_file_image)[0:samples[0]])
        test_images = set(list(list_file_image)[samples[0]:samples[0] + samples[1]])
        validate_images = set(list(list_file_image)[samples[0] + samples[1]:
                                                samples[0] + samples[1] + samples[2]])
        return train_images, test_images, validate_images


def form_samples(list_images, dir_sample):
    """Функция формирования выборок.

    :param list_images: лист с изображениями.
    :param dir_sample: папка сохранения (train, test, validate).
    """
    i = 1
    list_images_txt = []
    os.makedirs(os.path.join(save_path, dir_sample), exist_ok=True)
    for filename_image in list_images:
        current_filename = os.path.splitext(filename_image)[0]
        extension_file = os.path.splitext(filename_image)[-1]
        new_filename = '{}{}'.format(i, extension_file)
        shutil.move(os.path.join(path_to_dataset, '{}{}'.format(current_filename, extension_file)),
                    os.path.join(save_path, dir_sample, new_filename))
        shutil.move(os.path.join(path_to_dataset, '{}.txt'.format(current_filename)),
                    os.path.join(save_path, dir_sample, '{}.txt'.format(i)))
        list_images_txt.append('{}\n'.format(
            os.path.join(save_path, dir_sample, str(i) + extension_file)))
        i += 1
    write_txt_file(list_images_txt, save_path, dir_sample)


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
    samplings = create_lists_sampling(samples)
    if not samplings:
        return
    print("INFO: The script started it's work!")
    if len(samples) == 2:
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
    finish_time = timedelta(seconds=(time.time() - start_time))
    print('===========================')
    print('INFO: The run time of the script took {}.'.format(finish_time))
    print('INFO: File generated {0}.txt - {1}/{0}.txt'.format('train', save_path))
    print('INFO: File generated {0}.txt - {1}/{0}.txt'.format('test', save_path))
    if len(samples) == 3:
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
