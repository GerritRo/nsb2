from abc import abstractmethod

from nsb.core import Layer
from nsb.core.utils import reduce_rays

class Emitter(Layer):
    def __init__(self, config, N=1):               
        super().__init__(config, N, mode='forward')
        
    def forward(self, frame, rays):
        return self.emit(frame)
    
    def backward(self, frame, rays):
        return NotImplementedError
    
    @abstractmethod
    def emit(self, frame):
        return NotImplementedError

    def display(self, frame, rays):
        fig, ax = plt.subplots()
        ax.scatter(rays.az.deg, rays.alt.deg)
        return fig, ax

class DiffuseEmitter(Layer):
    def __init__(self, config, N=1):               
        super().__init__(config, N, mode='backward')
        
    def forward(self, frame, rays):
        return NotImplementedError
    
    @reduce_rays
    def backward(self, frame, rays):
        return self.evaluate(frame, rays)
    
    @abstractmethod
    def evaluate(self, frame, rays):
        return NotImplementedError

    def display(self, frame, rays):
        return NotImplementedError   