class Groundtruth:
    def __init__(self, path: str = '', boxes=None):
        if boxes is None:
            boxes = list()
        self.path = path
        self.boxes = boxes

    def __repr__(self):
        return self.path

    def __str__(self):
        return self.path
