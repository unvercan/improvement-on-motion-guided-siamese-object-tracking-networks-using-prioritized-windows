from __future__ import absolute_import, division

import cv2
import numpy as np
import torch.nn as nn


def initialize_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(image_path, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(image, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        image = cv2.cvtColor(image, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        out_size = (
            int(image.shape[1] * scale),
            int(image.shape[0] * scale))
        image = cv2.resize(image, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(image.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            image = cv2.rectangle(image, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, image)
        cv2.waitKey(delay)

    return image


def crop_and_resize(image, center, size, output_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - image.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        image = cv2.copyMakeBorder(
            image, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = image[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (output_size, output_size),
                       interpolation=interp)

    return patch


def calculate_apce(values):
    maximum = np.max(values)
    minimum = np.min(values)
    difference = maximum - minimum
    apce = (abs(difference) ** 2) / np.mean((values - minimum) ** 2)
    return apce


def calculate_p(values):
    values = np.array(values)
    maximum = np.max(values)
    minimum = np.min(values)
    difference = maximum - minimum
    return difference / np.mean((values / maximum))


def calculate_average(values, last_value_count=1):
    values = np.array(values)
    if len(values) == 0:
        return 0
    if len(values) <= last_value_count:
        last_values = values
    else:
        last_values = values[len(values) - last_value_count: len(values)]
    return np.mean(last_values)


def calculate_value_ratio(values, last_value_count=1):
    values = np.array(values)
    average = calculate_average(values=values, last_value_count=last_value_count)
    last_value = values[-1]
    return last_value / average


def normalize(values):
    values = np.array(values)
    maximum = np.max(values)
    minimum = np.min(values)
    difference = maximum - minimum
    return (values - minimum) / difference
