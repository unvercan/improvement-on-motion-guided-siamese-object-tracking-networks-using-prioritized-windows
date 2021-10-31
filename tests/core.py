from __future__ import absolute_import, print_function, unicode_literals

import numbers

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset


class AlexNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = dict()

    def forward(self, x):
        self.feature['conv1'] = self.conv1(x)
        self.feature['conv2'] = self.conv2(self.feature['conv1'])
        self.feature['conv3'] = self.conv3(self.feature['conv2'])
        self.feature['conv4'] = self.conv4(self.feature['conv3'])
        self.feature['conv5'] = self.conv5(self.feature['conv4'])
        return self.feature['conv5']


class AlexNetV1(AlexNetBase):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(num_features=96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), groups=2, padding=(0, 0)),
            nn.BatchNorm2d(num_features=256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(num_features=384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=2, padding=(0, 0)),
            nn.BatchNorm2d(num_features=384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=2, padding=(0, 0)),
        )


class Config(object):
    def __init__(self):
        # basic parameters
        self.output_scale = 0.001
        self.exemplar_size = 127
        self.instance_size = 255
        self.context = 0.5

        # inference parameters
        self.scale_count = 3
        self.scale_step = 1.0375
        self.scale_lr = 0.59
        self.scale_penalty = 0.9745
        self.window_influence = 0.176
        self.response_size = 17
        self.response_up = 16
        self.total_stride = 8

        # train parameters
        self.epoch_count = 50
        self.batch_size = 8
        self.worker_count = 32
        self.initial_lr = 1e-2
        self.ultimate_lr = 1e-5
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.r_pos = 16
        self.r_neg = 0

        # create hanning window
        self.upscale_size = (self.response_up * self.response_size)
        self.hanning_window = np.outer(np.hanning(self.upscale_size), np.hanning(self.upscale_size))
        self.hanning_window /= self.hanning_window.sum()

        # search scale factors
        self.scales = self.scale_step ** np.linspace(-(self.scale_count // 2), self.scale_count // 2, self.scale_count)


class Pair(Dataset):
    def __init__(self, sequences, transforms=None, pairs_per_sequence=1):
        super(Pair, self).__init__()
        self.sequences = sequences
        self.transforms = transforms
        self.pairs_per_sequence = pairs_per_sequence
        self.indices = np.random.permutation(len(sequences))
        self.return_meta = getattr(sequences, 'return_meta', False)

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]

        # get filename lists and annotations
        if self.return_meta:
            images, annotations, meta = self.sequences[index]
            visual_ratios = meta.get('cover', None)
        else:
            images, annotations = self.sequences[index][:2]
            visual_ratios = None

        # filter out noisy frames
        template = cv2.imread(images[0], cv2.IMREAD_COLOR)
        valid_indices = filter_indices(template, annotations, visual_ratios)
        if len(valid_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        random_z, random_x = sample_pair(valid_indices)
        z = cv2.imread(images[random_z], cv2.IMREAD_COLOR)
        x = cv2.imread(images[random_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        box_z = annotations[random_z]
        box_x = annotations[random_x]
        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            item = self.transforms(*item)
        return item

    def __len__(self):
        return len(self.indices) * self.pairs_per_sequence


def sample_pair(indices):
    n = len(indices)
    assert n > 0
    if n == 1:
        return indices[0], indices[0]
    elif n == 2:
        return indices[0], indices[1]
    else:
        for i in range(100):
            random_z, random_x = np.sort(np.random.choice(indices, 2, replace=False))
            if random_x - random_z < 100:
                break
        else:
            random_z = np.random.choice(indices)
            random_x = random_z
        return random_z, random_x


def filter_indices(template, annotations, visual_ratio=None):
    size = np.array(template.shape[1::-1])[np.newaxis, :]
    area = annotations[:, 2] * annotations[:, 3]

    # acceptance conditions
    condition_1 = area >= 20
    condition_2 = np.all(annotations[:, 2:] >= 20, axis=1)
    condition_3 = np.all(annotations[:, 2:] <= 500, axis=1)
    condition_4 = np.all((annotations[:, 2:] / size) >= 0.01, axis=1)
    condition_5 = np.all((annotations[:, 2:] / size) <= 0.5, axis=1)
    condition_6 = (annotations[:, 2] / np.maximum(1, annotations[:, 3])) >= 0.25
    condition_7 = (annotations[:, 2] / np.maximum(1, annotations[:, 3])) <= 4
    if visual_ratio is not None:
        condition_8 = (visual_ratio > max(1, visual_ratio.max() * 0.3))
    else:
        condition_8 = np.ones_like(condition_1)
    mask = np.logical_and.reduce((condition_1, condition_2, condition_3, condition_4,
                                  condition_5, condition_6, condition_7, condition_8))
    valid_indices = np.where(mask)[0]
    return valid_indices


class Correlation(nn.Module):
    def __init__(self, output_scale=0.001):
        super(Correlation, self).__init__()
        self.output_scale = output_scale

    def forward(self, z, x):
        return cross_correlation(z, x) * self.output_scale


def cross_correlation(z, x):
    # fast cross correlation
    batch_z = z.size(0)
    batch_x, channel_x, height_x, width_x = x.size()
    x = x.view(-1, batch_z * channel_x, height_x, width_x)
    output = functional.conv2d(x, z, groups=batch_z)
    output = output.view(batch_x, -1, output.size(-2), output.size(-1))
    return output


def init_weights(model, gain=1):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight, gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def read_image(image_path, color_code=cv2.COLOR_BGR2RGB):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_code is not None:
        image = cv2.cvtColor(image, color_code)
    return image


def show_image(image, boxes=None, box_format='ltwh', colors=None, thickness=3,
               fig_n=1, delay=1, visualize=True, color_code=cv2.COLOR_RGB2BGR):
    if color_code is not None:
        image = cv2.cvtColor(image, color_code)

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
        assert box_format in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_format == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(image.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
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


def crop_and_resize(image, center, size, output_size, border_type=cv2.BORDER_CONSTANT,
                    border=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((np.round(center - (size - 1) / 2), np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    paddings = np.concatenate((-corners[:2], corners[2:] - image.shape[:2]))
    number_of_paddings = max(0, int(paddings.max()))
    if number_of_paddings > 0:
        image = cv2.copyMakeBorder(image, number_of_paddings, number_of_paddings, number_of_paddings,
                                   number_of_paddings, border_type, value=border)

    # crop image patch
    corners = (corners + number_of_paddings).astype(int)
    patch = image[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (output_size, output_size), interpolation=interpolation)
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


def normalize(values):
    values = np.array(values)
    maximum = np.max(values)
    minimum = np.min(values)
    difference = maximum - minimum
    return (values - minimum) / difference


def log_sigmoid(x):
    # for x > 0: 0 - log(1 + exp(-x))
    # for x < 0: x - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + (0.5 * torch.clamp(x, min=0, max=0))


def log_minus_sigmoid(x):
    # for x > 0: -x - log(1 + exp(-x))
    # for x < 0:  0 - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + (0.5 * torch.clamp(x, min=0, max=0))


class BalancedLoss(nn.Module):
    def __init__(self, negative_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.negative_weight = negative_weight

    def forward(self, input_, target):
        positive_mask = (target == 1)
        negative_mask = (target == 0)
        number_of_positive = positive_mask.sum().float()
        number_of_negative = negative_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[positive_mask] = 1 / number_of_positive
        weight[negative_mask] = 1 / number_of_negative * self.negative_weight
        weight /= weight.sum()
        return functional.binary_cross_entropy_with_logits(input_, target, weight, reduction='sum')


class Net(nn.Module):
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class RandomStretch(object):
    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch

    def __call__(self, image):
        interpolation = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4
        ])
        scale = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        output_size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        return cv2.resize(image, output_size, interpolation=interpolation)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image):
        image_height, image_width = image.shape[:2]
        template_width, template_height = self.size
        i = round((image_height - template_height) / 2.)
        j = round((image_width - template_width) / 2.)

        number_of_paddings = max(0, -i, -j)
        if number_of_paddings > 0:
            average_color = np.mean(image, axis=(0, 1))
            image = cv2.copyMakeBorder(image, number_of_paddings, number_of_paddings, number_of_paddings,
                                       number_of_paddings, cv2.BORDER_CONSTANT, value=average_color)
            i += number_of_paddings
            j += number_of_paddings
        return image[i:i + template_height, j:j + template_width]


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image):
        image_height, image_width = image.shape[:2]
        template_width, template_height = self.size
        i = np.random.randint(0, image_height - template_height + 1)
        j = np.random.randint(0, image_width - template_width + 1)
        return image[i:i + template_height, j:j + template_width]


class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image).float().permute((2, 0, 1))


class SiamFCTransforms(object):
    def __init__(self, exemplar_size=127, instance_size=255, context=0.5):
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.context = context

        self.transforms_z = Compose([
            RandomStretch(),
            CenterCrop(instance_size - 8),
            RandomCrop(instance_size - 2 * 8),
            CenterCrop(exemplar_size),
            ToTensor()
        ])
        self.transforms_x = Compose([
            RandomStretch(),
            CenterCrop(instance_size - 8),
            RandomCrop(instance_size - 2 * 8),
            ToTensor()
        ])

    def __call__(self, z, x, box_z, box_x):
        z = self._crop(z, box_z, self.instance_size)
        x = self._crop(x, box_x, self.instance_size)
        z = self.transforms_z(z)
        x = self.transforms_x(x)
        return z, x

    def _crop(self, img, box, out_size):
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]
        ], dtype=np.float32)
        center, shape = box[:2], box[2:]

        context = self.context * np.sum(shape)
        size = np.sqrt(np.prod(shape + context))
        size *= (out_size / self.exemplar_size)

        average_color = np.mean(img, axis=(0, 1), dtype=float)
        interpolation = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4
        ])
        patch = crop_and_resize(img, center, size, out_size, border=average_color, interpolation=interpolation)
        return patch
