from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.optim as optimizer
from got10k.trackers import Tracker
from torch.optim.lr_scheduler import ExponentialLR

from siamfc.backbones import AlexNetV1
from siamfc.config import Config
from siamfc.heads import Correlation
from siamfc.helpers import show_image, read_image, crop_and_resize, initialize_weights
from siamfc.losses import BalancedLoss
from siamfc.network import Network

__all__ = ['CombinedSiam']


class CombinedSiam(Tracker):
    def __init__(self, weight_path=None):
        super(CombinedSiam, self).__init__('CombinedSiam', True)
        self.config = Config()

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        backbone = AlexNetV1()
        head = Correlation(output_scale=self.config.output_scale)
        self.network = Network(
            backbone=backbone,
            head=head
        )
        initialize_weights(model=self.network)

        # load checkpoint if provided
        if weight_path is not None:
            weight = torch.load(
                f=weight_path,
                map_location=lambda storage, loc: storage
            )
            self.network.load_state_dict(state_dict=weight)
        self.network = self.network.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optimizer.SGD(
            params=self.network.parameters(),
            lr=self.config.initial_lr,
            weight_decay=self.config.weight_decay,
            momentum=self.config.momentum
        )

        # setup lr scheduler
        gamma = np.power(
            (self.config.ultimate_lr / self.config.initial_lr),
            (1.0 / self.config.epoch_count)
        )
        self.lr_scheduler = ExponentialLR(
            optimizer=self.optimizer,
            gamma=gamma
        )

        self.center = self.shape = self.average = np.array([0, 0])
        self.z_size = self.x_size = 0
        self.kernel = dict()

        self.layer_ratio = {
            'conv1': 0.2,
            'conv3': 0.2,
            'conv5': 0.6
        }

    @torch.no_grad()
    def init(self, image, box):
        # set to evaluation mode
        self.network.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array(
            [
                ((box[1] - 1) + ((box[3] - 1) / 2)),
                ((box[0] - 1) + ((box[2] - 1) / 2)),
                box[3],
                box[2]
            ],
            dtype=np.float32
        )
        self.center, self.shape = box[:2], box[2:]

        # exemplar and search sizes
        context = self.config.context * np.sum(self.shape)
        self.z_size = np.sqrt(np.prod(self.shape + context))
        self.x_size = self.z_size * (self.config.instance_size / self.config.exemplar_size)

        # exemplar image
        self.average = np.mean(image, axis=(0, 1))
        z = crop_and_resize(
            image=image,
            center=self.center,
            size=self.z_size,
            output_size=self.config.exemplar_size,
            border_value=self.average
        )

        # exemplar features
        z = torch.from_numpy(z).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.network.backbone(z)

        self.kernel = {
            'conv1': self.network.backbone.feature['conv1'],
            'conv3': self.network.backbone.feature['conv3'],
            'conv5': self.network.backbone.feature['conv5']
        }

        search_window = np.array(
            [
                round((self.center[1] + 1) - (self.z_size / 2)),
                round((self.center[0] + 1) - (self.z_size / 2)),
                round(self.z_size),
                round(self.z_size)
            ]
        )

        return search_window

    @torch.no_grad()
    def update(self, image):
        # set to evaluation mode
        self.network.eval()

        # search images
        x_siam = [
            crop_and_resize(
                image=image,
                center=self.center,
                size=(self.x_size * scale),
                output_size=self.config.instance_size,
                border_value=self.average
            )
            for scale in self.config.scales
        ]
        x_siam = np.stack(x_siam, axis=0)
        x_siam = torch.from_numpy(x_siam).to(self.device).permute(0, 3, 1, 2).float()

        self.network.backbone(x_siam)
        x_siam = {
            'conv1': self.network.backbone.feature['conv1'],
            'conv3': self.network.backbone.feature['conv3'],
            'conv5': self.network.backbone.feature['conv5']
        }

        # responses
        responses_siam = dict()
        for layer in x_siam.keys():
            responses_siam[layer] = self.network.head(self.kernel[layer], x_siam[layer])
            responses_siam[layer] = responses_siam[layer].squeeze(1).to(self.device).numpy()

            # upsample responses and penalize scale changes
            responses_siam[layer] = np.stack(
                [
                    cv2.resize(
                        src=response,
                        dsize=(self.config.upscale_size, self.config.upscale_size),
                        interpolation=cv2.INTER_CUBIC
                    )
                    for response in responses_siam[layer]
                ]
            )
            responses_siam[layer][:self.config.scale_count // 2] *= self.config.scale_penalty
            responses_siam[layer][self.config.scale_count // 2 + 1:] *= self.config.scale_penalty

        # peak scale
        scale_siam_id = np.argmax(np.amax(responses_siam['conv5'], axis=(1, 2)))

        for layer in responses_siam.keys():
            # peak scale layer
            scale_siam_id_layer = np.argmax(np.amax(responses_siam[layer], axis=(1, 2)))

            # peak location
            responses_siam[layer] = responses_siam[layer][scale_siam_id_layer]
            responses_siam[layer] -= responses_siam[layer].min()
            responses_siam[layer] /= responses_siam[layer].sum() + 1e-16
            responses_siam[layer] = ((1 - self.config.window_influence) * responses_siam[layer]) + \
                                    (self.config.window_influence * self.config.hanning_window)

        # combine response
        response_siam = (responses_siam['conv1'] * self.layer_ratio['conv1']) + \
                        (responses_siam['conv3'] * self.layer_ratio['conv3']) + \
                        (responses_siam['conv5'] * self.layer_ratio['conv5'])

        # calculate peak
        # peak_siam = np.max(response_siam)

        # peak scale
        scale_siam = self.config.scales[scale_siam_id]

        search_window = np.array(
            [
                round((self.center[1] + 1) - ((self.x_size * scale_siam) / 2)),
                round((self.center[0] + 1) - ((self.x_size * scale_siam) / 2)),
                round(self.x_size * scale_siam),
                round(self.x_size * scale_siam)
            ]
        )

        # peak location
        location_siam = np.array(np.unravel_index(response_siam.argmax(), response_siam.shape))

        # locate target center
        displacement_in_response = location_siam - ((self.config.upscale_size - 1) / 2)
        displacement_in_instance = displacement_in_response * (self.config.total_stride / self.config.response_up)
        displacement_in_image = (displacement_in_instance * self.x_size) * (scale_siam / self.config.instance_size)
        self.center += displacement_in_image

        # update target size
        scale_ratio_siam = ((1 - self.config.scale_lr) * 1.0) + (self.config.scale_lr * scale_siam)
        self.shape *= scale_ratio_siam
        self.z_size *= scale_ratio_siam
        self.x_size *= scale_ratio_siam

        # return 1-indexed and left-top based bounding box
        box = np.array(
            [
                ((self.center[1] + 1) - ((self.shape[1] - 1) / 2)),
                ((self.center[0] + 1) - ((self.shape[0] - 1) / 2)),
                self.shape[1],
                self.shape[0]
            ]
        )

        return box, search_window

    def track(self, image_paths, box, visualize=False, annotations=None):
        frame_count = len(image_paths)
        boxes = np.zeros((frame_count, 4))
        search_windows = np.zeros((frame_count, 4))
        boxes[0] = box

        for frame_index, image_path in enumerate(image_paths):
            image = read_image(image_path=image_path)

            if frame_index == 0:
                search_windows[frame_index, :] = self.init(
                    image=image,
                    box=box
                )
            else:
                boxes[frame_index, :], search_window = self.update(image=image)
                if search_window is not None:
                    search_windows[frame_index, :] = search_window
                else:
                    search_windows[frame_index] = None

            if visualize:
                box_list = boxes[frame_index, :]
                if annotations is not None:
                    if search_windows[frame_index] is not None:
                        box_list = [
                            boxes[frame_index, :],
                            annotations[frame_index, :],
                            search_windows[frame_index, :]
                        ]
                    else:
                        box_list = [
                            boxes[frame_index, :],
                            annotations[frame_index, :]
                        ]
                show_image(
                    image=image,
                    boxes=box_list
                )

        return boxes
