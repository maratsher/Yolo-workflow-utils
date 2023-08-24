"""
Скрипт аугментации данных.
Включает в себя следующие опции аугментации:
    motion_blur
    darkness
    brightness
    noise
    contrast
    shadow (для darknet bbox)
    shadow class
    flip

Использование: python3 augmentation_dataset.py -d %path_to_dataset%
-s %path_to_save_aug_dataset% -cont -dark -random
"""
import os
import re
import shutil
import random
from os import path, makedirs
from argparse import ArgumentParser

import cv2
import tqdm
import numpy as np
import imgaug.augmenters as iaa
from imgaug import BoundingBoxesOnImage

from utils import collection_data, convert_labels, transform_coordinates


class Augmentation:
    """
    Класс Аугментации.
    """
    @classmethod
    def random_param(cls, aug_param):
        """Функция возвращает рандомное число
        для одного из параметра аугментации.

        :param aug_param: параметр аугментации.
        :return: рандомное число.
        """
        return aug_param[random.randint(0, len(aug_param) - 1)]

    @staticmethod
    def flip_image_and_save_coords_bbox(image, boxes):
        """Функция записывает текущие координаты bbox в класс BoundingBox
         для последующейработы с ними.

        :param image: изображение (массив).
        :param boxes: координаты bbox.
        """
        list_bboxes = []
        for bbox in boxes:
            if len(bbox) > 4:
                box = iaa.ia.BoundingBox(x1=bbox[1], x2=bbox[3],
                                         y1=bbox[2], y2=bbox[4],
                                         label=str(bbox[0]))
            else:
                box = iaa.ia.BoundingBox(x1=bbox[0], x2=bbox[2],
                                         y1=bbox[1], y2=bbox[3],)
            list_bboxes.append(box)
        bbs = BoundingBoxesOnImage(list_bboxes, shape=image.shape)

        return bbs

    @staticmethod
    def copy_label_and_image(path_to_dataset, path_to_save_aug, filename, image_aug):
        """Функция копирует аугментированное изображение в папку сохранения.

        :param path_to_dataset: путь до датасета.
        :param path_to_save_aug: путь сохранения аугментированных данных.
        :param filename: название изображения.
        :param image_aug: аугментированное изображение.
        """
        i = 0
        makedirs(path_to_save_aug, exist_ok=True)
        old_filename = os.path.basename(filename)
        while path.exists(path.join(path_to_save_aug, os.path.basename(filename))):
            i += 1
            filename = '{}_{}{}'.format(path.splitext(os.path.basename(filename))[0], i,
                                        path.splitext(os.path.basename(filename))[-1])
        cv2.imwrite(path.join(path_to_save_aug, os.path.basename(filename)), image_aug)
        shutil.copy(path.join(path_to_dataset,
                              '{}.txt'.format(path.splitext(old_filename)[0])),
                    path.join(path_to_save_aug,
                              '{}.txt'.format(path.splitext(os.path.basename(filename))[0])))

    @staticmethod
    def flipped_images(path_to_save_aug, filename, image, image_aug):
        """Функция отражет изображение по горизонтали и сохраняет в папку.

        :param path_to_save_aug: путь до датасета.
        :param filename: название изображения.
        :param image: изображение.
        :param image_aug: аугментированное изображение.
        """
        i = 0
        makedirs(path_to_save_aug, exist_ok=True)
        old_filename = filename
        while path.exists(path.join(path_to_save_aug, filename)):
            i += 1
            filename = '{}_{}{}'.format(path.splitext(filename)[0], i,
                                        path.splitext(filename)[-1])
        image = np.fliplr(image)
        cv2.imwrite(path.join(path_to_save_aug, filename), image)
        flip_boxes_list = convert_labels(image, image_aug[1],
                                         class_in_box=True,
                                         edited_box=True)
        new_boxes = [re.sub(r'\(|\)|,', '', str(bbox)) + '\n'
                     for bbox in flip_boxes_list]
        with open(path.join(
                path_to_save_aug,
                '{}.txt'.format(path.splitext(old_filename)[0])),  'w') as f:
            f.writelines(new_boxes)


    def augmentations_options(self, image, filename, path_to_save_aug,
                              motion_blur, darkness, brightness,
                              noise, contrast, shadow, flipped, random_aug):
        """Функция, содержащая основные параметры аугментации.

        :param image: изображение.
        :param filename: названия изображения.
        :param motion_blur: размытие в движении.
        :param darkness: затемнение.
        :param brightness: засветление.
        :param noise: гауссово размытие.
        :param contrast: контрастность.
        :param flipped: отражение по горизонтали.
        :param shadow: тень.
        :param random_aug: параметр, применять рандомно один из параметров
        аугментации или поочередно сохранять к одному изображению несколько
        аугментированных.
        """
        aug_options = [motion_blur, darkness, brightness,
                       noise, contrast, shadow, flipped]
        if random_aug:
            while aug_options.count(True) != 1:
                index_trues = [i for i, aug in enumerate(aug_options) if aug is True]
                random_index_true = random.choice(index_trues)
                aug_options[random_index_true] = False
        if aug_options[0]:
            motion_blur_aug = iaa.Sequential(
                [
                    iaa.MotionBlur(k=range(5, 10),
                                   angle=0,
                                   random_state=True),
                ], random_order=True)
            self.copy_label_and_image(path_to_dataset, path_to_save_aug,
                                      filename, motion_blur_aug(images=image))

        if aug_options[1]:
            mul_dark = [x * (random.randint(5, 10) * 0.01) for x in range(2, 10)]
            random_parameter = self.random_param(mul_dark)
            darkness_aug = iaa.Sequential(
                [
                    iaa.arithmetic.Multiply(mul=random_parameter)
                ], random_order=True)

            self.copy_label_and_image(path_to_dataset, path_to_save_aug,
                                      filename, darkness_aug(images=image))

        if aug_options[2]:
            mul_bright = [x * (random.randint(5, 10) * 0.1) for x in range(1, 4)]
            random_parameter = self.random_param(mul_bright)
            brightness_aug = iaa.Sequential(
                [
                    iaa.arithmetic.Multiply(mul=random_parameter),
                ], random_order=True)

            self.copy_label_and_image(path_to_dataset, path_to_save_aug,
                                      filename, brightness_aug(images=image))

        if aug_options[3]:
            coefficient_gaussian = [x * (random.randint(1, 5) * 0.01) for x in range(1, 10)]
            random_parameter = self.random_param(coefficient_gaussian)
            gaussian_noise = iaa.Sequential(
                [
                    iaa.AdditiveGaussianNoise(scale=random_parameter * 255,
                                              random_state=True)
                ], random_order=True)

            self.copy_label_and_image(path_to_dataset, path_to_save_aug,
                                      filename, gaussian_noise(images=image))

        if aug_options[4]:
            clip_limit = [x for x in range(1, 20)]
            random_parameter = self.random_param(clip_limit)
            low_contrast_aug = iaa.Sequential(
                [
                    iaa.AllChannelsCLAHE(clip_limit=random_parameter),
                ], random_order=True)

            self.copy_label_and_image(path_to_dataset, path_to_save_aug,
                                      filename, low_contrast_aug(images=image))

        if aug_options[5]:
            bboxes = transform_coordinates(path_to_dataset,
                                           image,
                                           filename,
                                           add_class_in_box=True)
            for bbox in bboxes:
                if bbox[0] in shadow_class:
                    cutted_img = image[bbox[2]:bbox[4], bbox[1]:bbox[3]]
                    random_no_of_shadows = [x for x in range(5, 10)]
                    no_of_shadows = self.random_param(random_no_of_shadows)
                    random_shadow_dimension = [x for x in range(3, 10)]
                    shadow_dimension = self.random_param(random_shadow_dimension)
                    x1 = 0
                    y1 = cutted_img.shape[0] // 2
                    x2 = cutted_img.shape[1]
                    y2 = cutted_img.shape[0]
                    image_colorsapce_hsl = cv2.cvtColor(cutted_img, cv2.COLOR_RGB2HLS)
                    mask = np.zeros_like(cutted_img)
                    vertices_list = []
                    for index in range(no_of_shadows):
                        vertex = []
                        for dimensions in range(shadow_dimension):
                            vertex.append(
                                (random.randint(x1, x2), random.randint(y1, y2)))
                        vertices = np.array([vertex], dtype=np.int32)
                        vertices_list.append(vertices)
                    for vertices in vertices_list:
                        cv2.fillPoly(mask, vertices, 255)
                    random_shadow_coefficient = [x * (random.randint(9, 11) * 0.01)
                                                 for x in range(3, 5)]
                    shadow_coefficient = self.random_param(random_shadow_coefficient)
                    image_colorsapce_hsl[:, :, 1][mask[:, :, 0] == 255] = \
                        image_colorsapce_hsl[:, :, 1][
                            mask[:, :, 0] == 255] * shadow_coefficient
                    image_shadow = cv2.cvtColor(image_colorsapce_hsl,
                                                cv2.COLOR_HLS2RGB)
                    y_offset, x_offset = bbox[2], bbox[1]
                    image[y_offset:y_offset + image_shadow.shape[0],
                    x_offset:x_offset + image_shadow.shape[1]] = image_shadow

            self.copy_label_and_image(path_to_dataset, path_to_save_aug,
                                      filename, image)

        if aug_options[6]:
            flip = iaa.Sequential([iaa.Fliplr(1)])
            file_img = path.join(path_to_dataset, '{}.txt'.format(
                path.basename(filename)[:-4]))
            boxes = transform_coordinates(path_to_dataset,
                                          image,
                                          file_img,
                                          add_class_in_box=True)
            list_bboxes = self.flip_image_and_save_coords_bbox(image, boxes)
            self.flipped_images(path_to_save_aug, filename, image,
                                flip(images=image, bounding_boxes=list_bboxes))

    def augmentations_images(self, path_to_dataset, path_to_save_aug):
        """Функция берет в цикле изображения из датасета и применяет к ним
        параметры аугментации.
        """
        print('INFO: Sampling augmentation started!')
        list_file_image, list_file_text = collection_data(path_to_dataset)
        for filename in tqdm.tqdm(list_file_image):
            image = cv2.imread(path.join(path_to_dataset, os.path.basename(filename)))
            filename = os.path.basename(filename)
            self.augmentations_options(image=image,
                                       filename=filename,
                                       path_to_save_aug=path_to_save_aug,
                                       motion_blur=motion_blur,
                                       darkness=darkness,
                                       brightness=brightness,
                                       noise=noise,
                                       contrast=contrast,
                                       shadow=shadow,
                                       flipped=flip,
                                       random_aug=random_aug)
        print('INFO: Sample augmentation complete!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir_to_dataset",
                        required=False, help="The path to the dataset")
    parser.add_argument("-s", "--save_to_augmentation",
                        required=False, help="The path to the save augmentation dataset")
    parser.add_argument("-mb", "--motion_blur",
                        required=False, help="Add motion blur",
                        default=False,
                        action='store_true')
    parser.add_argument("-dark", "--darkness",
                        required=False, help="Add dimming to Image",
                        default=False,
                        action='store_true')
    parser.add_argument("-bright", "--brightness",
                        required=False, help="Add image overlay",
                        default=False,
                        action='store_true')
    parser.add_argument("-noise", "--noise",
                        required=False, help="Adds Gaussian Blur",
                        default=False,
                        action='store_true')
    parser.add_argument("-cont", "--contrast",
                        required=False, help="Adds contrast to Clahe",
                        default=False,
                        action='store_true')
    parser.add_argument("-sh", "--shadow",
                        required=False, help="Adds a shadow to the bounding box",
                        default=False, action='store_true')
    parser.add_argument("-sh_c", "--shadow_class",
                        required=False,
                        help="What class to apply shadow augmentation to",
                        nargs='+', type=int)
    parser.add_argument("-flip", "--flipped",
                        required=False, help="Flipped images",
                        default=False, action='store_true')
    parser.add_argument("-random", "--random_param",
                        required=False, help="Applies one of the selected "
                                             "augmentation options to the image",
                        default=False, action='store_true')

    args = vars(parser.parse_args())

    path_to_dataset = args['dir_to_dataset']
    path_to_save_aug = args['save_to_augmentation']
    motion_blur = bool(args['motion_blur'])
    darkness = bool(args['darkness'])
    brightness = bool(args['brightness'])
    noise = bool(args['noise'])
    contrast = bool(args['contrast'])
    shadow = bool(args['shadow'])
    shadow_class = args['shadow_class'] if args['shadow_class'] else range(80)
    flip = bool(args['flipped'])
    random_aug = bool(args['random_param'])

    augmentation = Augmentation()
    augmentation.augmentations_images(path_to_dataset,
                                      path_to_save_aug)
