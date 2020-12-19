from . import IBaseDev
from torch.cuda import empty_cache


class IBaseDevTorch(IBaseDev):
    """PyTorch specific base interface for development."""
    
    def train(self, *args, **kwargs):
        empty_cache()
        return super(IBaseDevTorch, self).train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        empty_cache()
        return super(IBaseDevTorch, self).eval(*args, **kwargs)
