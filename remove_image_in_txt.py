"""
Скрипт удаления изображений из файлов [train.txt, text.txt, ..]
"""
import os
from argparse import ArgumentParser

from utils import return_path_to_file, return_format_image


def remove_image_in_text_file():
    """Функция очистки путей до проблемных изображений на основании
    подаваемого файла bad_list. """
    with open(path_to_text_file, 'r') as f:
        text_file = f.readlines()

    with open(path_to_bad_list, 'r') as f:
        bad_list = f.readlines()

    new_bad_list = []
    for bad_line in bad_list:
        path_to_file = return_path_to_file(bad_line)
        if os.path.isfile(bad_line[:-1]):
            format_image = os.path.splitext(bad_line)[-1][:-1]
        else:
            format_image = return_format_image(bad_line)
        new_bad_list.append('{}{}\n'.format(os.path.splitext(path_to_file)[0],
                                            format_image))

    new_text_list = []
    for line_text in text_file:
        if line_text not in new_bad_list:
            new_text_list.append(line_text)

    with open(path_to_text_file, 'w') as f:
        f.writelines(new_text_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-b",
                        "--bad_list",
                        required=False,
                        help="The path to the bad list")
    parser.add_argument("-t",
                        "--text_file",
                        required=False,
                        help="The path to the text file")

    args = vars(parser.parse_args())
    path_to_bad_list = args['bad_list']
    path_to_text_file = args['text_file']

    remove_image_in_text_file()
