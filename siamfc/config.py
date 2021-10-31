import numpy as np


# default parameters
class Config:
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
        self.hanning_window = np.outer(
            np.hanning(self.upscale_size),
            np.hanning(self.upscale_size)
        )
        self.hanning_window /= self.hanning_window.sum()

        # search scale factors
        self.scales = self.scale_step ** np.linspace(
            -(self.scale_count // 2),
            self.scale_count // 2,
            self.scale_count
        )
