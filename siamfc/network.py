from torch import nn


class Network(nn.Module):

    def __init__(self, backbone, head):
        super(Network, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)
