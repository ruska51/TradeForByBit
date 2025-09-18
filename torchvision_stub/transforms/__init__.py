class Compose:
    def __init__(self, t=None):
        pass
    def __call__(self, x):
        return x

class Resize:
    def __init__(self, size):
        pass
    def __call__(self, x):
        return x

class ToTensor:
    def __call__(self, x):
        return x
