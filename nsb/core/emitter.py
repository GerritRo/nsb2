from abc import abstractmethod

from nsb.core import Layer
from nsb.core.utils import reduce_rays


class Emitter(Layer):
    """
    A layer responsible for emitting forward rays based on the given observational frame

    Parameters
    ----------
    config : dictionary
        Dictionary informing the parameters of the layer at compilation
    """
    def __init__(self, config):
        super().__init__(config, mode="forward")

    def forward(self, frame, rays):
        return self.emit(frame)

    def backward(self, frame, rays):
        return NotImplementedError

    @abstractmethod
    def emit(self, frame):
        return NotImplementedError


class Diffuse(Layer):
    """
    A layer responsible for evaluating backwards rays hitting a diffuse emission.

    Parameters
    ----------
    config : dictionary
        Dictionary informing the parameters of the layer at compilation
    """
    def __init__(self, config):
        super().__init__(config, mode="backward")

    def forward(self, frame, rays):
        return NotImplementedError

    def backward(self, frame, rays):
        return self.evaluate(frame, rays)

    @abstractmethod
    def evaluate(self, frame, rays):
        return NotImplementedError
