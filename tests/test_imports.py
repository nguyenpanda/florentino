import importlib
import sys

import pytest
from nguyenpanda.swan import Color

import florentino as flo

yellow = Color['y']
magenta = Color['p']
reset = Color.reset


def test_print_all_submodules_and_classes():
    print('\n' + yellow + f'{str(" All modules in `florentino`"):=^50}' + reset, file=sys.stdout)
    for submodule_name in flo.__all__:
        submodule = importlib.import_module(f'florentino.{submodule_name}')
        print(magenta + submodule.__name__ + reset, file=sys.stdout)
        for class_name in submodule.__all__:
            print(f'\t - {class_name}', file=sys.stdout)
    print(yellow + 50 * '=' + reset, file=sys.stdout)


def test_import_florentino_submodules_and_classes():
    for submodule_name in flo.__all__:
        submodule = __import__(f'florentino.{submodule_name}', fromlist=[submodule_name])
        assert submodule.__name__ == f'florentino.{submodule_name}', f"Failed to import {submodule_name}"

        for class_name in submodule.__all__:
            cls = getattr(submodule, class_name, None)
            assert cls is not None, f"Class {class_name} not found in {submodule_name}"
            assert cls.__name__ == class_name, f"Class name {cls.__name__} does not match expected {class_name}"

        del submodule


if __name__ == '__main__':
    pytest.main([__file__])
