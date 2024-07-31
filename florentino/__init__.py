from .__version__ import __version__

VERSION = __version__

__florentino_submodules__ = {
    'linear_model',
    'nn',
}

__universal_function__: set = set()

__all__ = list(
    __florentino_submodules__ |
    __universal_function__
)
