"""
Скрипт проверки наличия файла разметки к изображению.
"""
from argparse import ArgumentParser
from os import path, getcwd

from tqdm import tqdm

from utils import collection_data


def check_images_for_label(path_to_dataset):
    """Функция поиска файлов разметки к изображению.
    Если к изображению отсутствует файл разметки, то происходит
    запись в bad.list.

    :param :path_to_dataset: путь до датасета.
    """
    list_file_image, list_file_text = collection_data(path_to_dataset)
    bad_list = []
    if not list_file_image:
        print('INFO: There is no data to process in the folder.')
    else:
        for filename in tqdm(list_file_image):
            if path.splitext(filename)[0] in list_file_text:
                list_file_text.remove(path.splitext(filename)[0])
            else:
                bad_list.append('{}\n'.format(path.join(path_to_dataset, filename)))
        if bad_list:
            with open('bad.list', 'w') as f:
                f.writelines(bad_list)
            print('Bad list formed: {}'.format(path.join(getcwd(), 'bad.list')))
        else:
            print('INFO: No warnings found.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d",
                        "--dir_to_dataset",
                        required=False,
                        help="The path to the dataset")

    args = vars(parser.parse_args())
    path_to_dataset = args['dir_to_dataset']

    check_images_for_label(path_to_dataset)
