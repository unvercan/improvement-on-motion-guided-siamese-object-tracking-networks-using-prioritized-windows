class Sequence:
    def __init__(self, name: str = '', images=None, groundtruths=None, attributes=None):
        if attributes is None:
            attributes = list()
        if groundtruths is None:
            groundtruths = list()
        if images is None:
            images = list()
        self.name = name
        self.images = images
        self.groundtruths = groundtruths
        self.attributes = attributes

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
