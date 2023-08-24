"""
Скрипт удаления данных из bad листов.
"""
import os
from argparse import ArgumentParser

from tqdm import tqdm

from utils import return_path_to_file, return_format_image


def delete_bad_list_data():
    """Функция удаляющая изображения исходя из данных, находящихся в
    файле bad list."""
    print("INFO: The script started it's work!")
    with open(path_to_bad_list, 'r') as f:
        bad_list = f.readlines()
    for line in tqdm(bad_list):
        path_to_file = return_path_to_file(line)
        format_image = return_format_image(line)
        if path_to_file:
            try:
                os.remove(path_to_file)
            except FileNotFoundError:
                pass
            try:
                os.remove('{}{}'.format(os.path.splitext(path_to_file)[0],
                                        format_image))
            except FileNotFoundError:
                pass
    print('INFO: Script completed!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-b",
                        "--bad_list",
                        required=False,
                        help="The path to the bad list")

    args = vars(parser.parse_args())
    path_to_bad_list = args['bad_list']
    delete_bad_list_data()
