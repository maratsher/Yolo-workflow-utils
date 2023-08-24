from ctypes import *

import cv2
import yaml
import numpy as np


with open('config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class BOX(Structure):
    _fields_ = [('x', c_float),
                ('y', c_float),
                ('w', c_float),
                ('h', c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [('w', c_int),
                ('h', c_int),
                ('c', c_int),
                ('data', POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [('classes', c_int),
                ('names', POINTER(c_char_p))]


lib = CDLL(config['darknet']['libdarknet'], RTLD_GLOBAL)

get_network_width = lib.network_width
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int

get_network_height = lib.network_height
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float,
                              POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(new_arr):
    new_arr = new_arr.transpose(2, 0, 1)
    c, h, w = new_arr.shape

    new_arr = np.ascontiguousarray(new_arr.flat, dtype=np.float32) / 255.0
    im = IMAGE(w, h, c, new_arr.ctypes.data_as(POINTER(c_float)))

    # need to return arr to avoid python freeing memory
    return im, new_arr


def net_size_image(config_path, weights_path):
    net = load_net(config_path.encode('utf-8'),
                   weights_path.encode('utf-8'), 0)
    net_w, net_h = get_network_width(net), get_network_height(net)

    return net_w, net_h


def net_classes_names(data_path):
    meta = load_meta(data_path.encode('utf-8'))
    names = []
    for i in range(meta.classes):
        names.append(meta.names[i].decode('utf-8'))
    return meta.classes, names


class Darknet:
    def __init__(self, config_path, weights_path, meta_path):
        self.meta = load_meta(meta_path.encode('utf-8'))
        self.net = load_net(config_path.encode('utf-8'),
                            weights_path.encode('utf-8'), 0)

        # на самом деле net_w = net_h
        self.net_w = get_network_width(self.net)
        self.net_h = get_network_height(self.net)
        _, self.names = net_classes_names(meta_path)


class YoloV3(Darknet):
    def __init__(self, config_path, weights_path, meta_path):
        super().__init__(config_path, weights_path, meta_path)

    def get_boxes(self, bgr_image, classes, thresh=.5, hier_thresh=.5, nms=.45, letter_box=False):
        frame_h, frame_w, _ = bgr_image.shape

        # на самом деле net_w = net_h
        net_w, net_h = get_network_width(self.net), get_network_height(self.net)
        if letter_box:
            # вычисляем все для ресайза с сохранием пропорций
            if net_w / frame_w < net_h / frame_h:
                new_size = (net_w, int(net_w / frame_w * frame_h))
            else:
                new_size = (int(net_h / frame_h * frame_w), net_h)

            # вычисляем отступы, которые необходимо закрасить серым,
            # чтобы получить квадратное изображение
            top = (net_h - new_size[1]) // 2
            bottom = net_h - top - new_size[1]
            left = (net_w - new_size[0]) // 2
            right = net_w - left - new_size[0]

            prepared_image = cv2.resize(bgr_image, new_size)
            prepared_image = cv2.copyMakeBorder(
                prepared_image, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(127, 127, 127))
        else:
            prepared_image = cv2.resize(bgr_image, (net_w, net_h))

        prepared_image = cv2.cvtColor(prepared_image, cv2.COLOR_BGR2RGB)
        prepared_image, _ = array_to_image(prepared_image)

        predict_image(self.net, prepared_image)

        detections_count = c_int(0)
        p_detections_count = pointer(detections_count)

        detections = get_network_boxes(
            self.net, net_w, net_h,
            thresh, hier_thresh, None, 0, p_detections_count, 0
        )

        detections_count = p_detections_count[0]

        if nms:
            do_nms_sort(detections, detections_count, self.meta.classes, nms)

        # вычисляем параметры для рескейла коробок
        if letter_box:
            rescale_w = frame_w / new_size[0]
            rescale_h = frame_h / new_size[1]
        res = []
        for j in range(detections_count):
            for i in range(classes):
                if detections[j].prob[i]:
                    # name = self.meta.names[i].decode()

                    box = detections[j].bbox
                    # prob = detections[j].prob[i]
                    if letter_box:
                        box.x = (box.x - left) * rescale_w
                        box.y = (box.y - top) * rescale_h
                        box.w *= rescale_w
                        box.h *= rescale_h

                    x1 = box.x - box.w / 2
                    y1 = box.y - box.h / 2
                    x2 = box.x + box.w / 2
                    y2 = box.y + box.h / 2

                    x1 = int(x1 if x1 > 0 else 0)
                    y1 = int(y1 if y1 > 0 else 0)
                    x2 = int(x2 if x2 < frame_w else frame_w)
                    y2 = int(y2 if y2 < frame_h else frame_h)

                    res.append((i, x1, y1, x2, y2))

        free_detections(detections, detections_count)
        return res


class DarknetClassifier(Darknet):
    def classify_image(self, image: np.ndarray):
        prepared_image, _ = array_to_image(image)
        out = predict_image(self.net, prepared_image)
        res = []
        for i in range(self.meta.classes):
            res.append((self.meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res
