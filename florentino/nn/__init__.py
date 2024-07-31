import inspect

from .activation import Activation
from .dense import Dense
from .loss import *
from .network import Network
from .softmax import Softmax

__core_component__ = {
    'Activation',
    'Dense',
    'Network',
    'Softmax'
}


def get_classes(module) -> set[str]:
    """List all classes in a given module."""
    return set(name for name, obj in inspect.getmembers(module, inspect.isclass))


__all__ = list(
    get_classes(loss) |
    __core_component__
)
