import os
import pathlib

from typing_extensions import Optional, List, Tuple

from .activation import *
from .dense import Dense
from .layer import Layer
from .loss import *
from .softmax import Softmax
from .utillity import print_batch


def group(lst):
    return list(zip(lst[::2], lst[1::2]))


class Network:

    def __init__(self, sequential: List[Layer], loss_func: Loss):
        # @formatter:off
        self.sequential: List[Layer]        = sequential
        self.loss: Loss                     = loss_func
        self.classification                 = True
        self.x_train: Optional[np.ndarray]  = None
        self.y_train: Optional[np.ndarray]  = None
        self.n: Optional[np.ndarray]        = None
        self.m: Optional[np.ndarray]        = None
        # @formatter:on

    def fit(self, x_train, y_train, classification=True):
        self.x_train = x_train
        if classification:
            self.y_train = self._one_hot(y_train)
        else:
            self.y_train = y_train
            self.classification = False
        self.n, self.m = x_train.shape

    def predict(self, x) -> np.ndarray:
        for layer in self.sequential:
            x = layer.validate(x)
        return x

    def cost(self, x=None, y=None) -> float:
        if x is None and y is None:
            return self._cost(self.x_train, self.y_train)
        if x is None or y is None:
            raise ValueError()
        return self._cost(x, y)

    def accuracy(self, x=None, y=None) -> float:
        if x is None and y is None:
            return self._accuracy(self.x_train, self.y_train)
        if x is None or y is None:
            raise ValueError()
        return self._accuracy(x, y)

    def train(self, epochs: int = 1000, alpha: float = 0.1, batches: Optional[int] = None,
              validation_data: Optional[List | Tuple] = None, verbose: bool = True):
        # @formatter:off
        if batches is None:
            batches     = 1
        batch_size      = self.m // batches
        remain          = self.m % batches
        batch_pad       = len(str(batches))
        epoch_pad       = len(str(epochs))
        alpha_pad       = len(str(epochs))
        header_width    = 84
        unit            = 'accuracy' if self.classification else 'error'
        error           = self.accuracy if self.classification else self.cost
        # @formatter:on

        if verbose:
            print('-' * header_width)
            print(f'| epochs = {epochs: <16} | batches = {batches: <16} | alpha = {alpha: <15} |')
            print('-' * header_width)
            print(
                f'| x_train.shape = {str(self.x_train.shape): <24} | y_train.shape = {str(self.y_train.shape): <21} |')
            print('-' * header_width)

        x_val = y_val = None
        if validation_data:
            if not (isinstance(validation_data, list) or isinstance(validation_data, tuple)):
                raise ValueError('validation_data should be a list or tuple')
            if len(validation_data) != 2:
                raise ValueError('validation_data must have 2 elements')
            x_val, y_val = validation_data
            print(f'| x_val.shape   = {str(x_val.shape): <24} | y_val.shape   = {str(y_val.shape): <21} |')
            print('-' * header_width)

        for e in range(1, epochs + 1):
            start = 0
            for b in range(1, batches + 1):
                end = start + batch_size
                self._train(start, end, alpha)
                start = end

                if remain > 0:
                    self._train(-remain, None, alpha)

                if verbose:
                    if batches == 1:
                        print(f'{e: >{epoch_pad}}/{epochs}, {unit}=\033[1;92m{error()}\033[0m')
                    else:
                        adding_string = f'{unit}=\033[1;92m{error():>7.5f}\033[0m'
                        if validation_data:
                            adding_string += f' | val_{unit}=\033[1;92m{self.accuracy(x_val, y_val):>7.5f}\033[0m '
                        print_batch(adding_string, e, epochs, epoch_pad, b, batches, batch_pad)

            if verbose and batches != 1:
                print()  # Reset '\r'

    def summary(self):
        total_para = 0
        total_layer = len(self.sequential) + 2
        network_type = 'classification' if self.classification else 'regression'
        input_nodes = f'InputLayer({self.sequential[0].pre_nodes})'
        output_nodes = f'OutputLayer({self.sequential[-1].nodes})'

        header = f"| {str('idx'):>4} | {str('Layer'):^62} | {str('Parameters'):>12} |"
        row_len = len(header)

        print('=' * row_len)
        print(header)
        print('=' * row_len)

        print(f"| {0:>4} | {input_nodes:>62} | {0:>12} |")
        for i, layer in enumerate(self.sequential, 1):
            total_para += len(layer)
            print(f"| {i:>4} | {str(layer):>62} | \033[1;92m{len(layer):>12}\033[0m |")
        print(f"| {total_layer - 1:>4} | {output_nodes:>62} | {0:>12} |")

        print('=' * row_len)
        print(f"| {' ':>4} | {total_layer:>62} | {total_para:>12} |")
        print('=' * row_len)
        print(f"| {str('Loss function: ') + str(self.loss):>40} || {str('Network type: ') + network_type:>40} |")
        print('=' * row_len)

    @staticmethod
    def load(directory: str | pathlib.Path, loss_func: Loss):
        directory_path = pathlib.Path(directory) if isinstance(directory, str) else directory

        files = sorted([file for file in list(os.listdir(directory_path)) if file.endswith('.npy')])

        if len(files) % 2 != 0:
            raise ValueError('Number of files must be even')

        # @formatter:off
        datas       = group([np.load(directory_path / file) for file in files])
        tokens      = group([file.removesuffix('.npy').split('_') for file in files])
        files       = group(files)
        # @formatter:on

        check_missing = 1
        sequential = []
        for i, token in enumerate(tokens):
            b = token[0]
            w = token[1]
            data = datas[i]

            if check_missing != int(b[0]):
                raise ValueError(f'Missing parameter b of layer {check_missing}')
            if check_missing != int(w[0]):
                raise ValueError(f'Missing parameter w of layer {check_missing}')
            check_missing += 1

            if b[1] != 'b':
                raise ValueError(f'Invalid parameter type: {b[1]}')
            if w[1] != 'w':
                raise ValueError(f'Invalid parameter type: {w[1]}')
            if b[2] != w[2]:
                raise ValueError(f'Invalid layer type: {b[2]}')

            data_b = data[0]
            data_w = data[1]
            nodes, pre_nodes = data_w.shape
            if data_b.shape[0] != data_w.shape[0]:
                raise ValueError(f'Invalid shape of data: {data_b.shape} and {data_w.shape}')

            if b[2] == 'Dense':
                if b[3] != w[3]:
                    raise ValueError(f'Invalid activation function: {b[3]}')
                temp = Dense(nodes, pre_nodes, activation=b[3])
                temp.b = data_b
                temp.w = data_w
                sequential.append(temp)
            elif b[2] == 'Softmax':
                temp = Softmax(nodes, pre_nodes)
                temp.b = data_b
                temp.w = data_w
                sequential.append(temp)
        return Network(sequential, loss_func)

    def save(self, directory_name: str, exist_ok=False):
        direct_path = pathlib.Path(directory_name)
        direct_path.mkdir(parents=True, exist_ok=exist_ok)

        for i, layer in enumerate(self.sequential, 1):
            file_name_w = f'{i}_w_'
            file_name_b = f'{i}_b_'

            layer_name = layer.__class__.__name__
            if isinstance(layer, Dense):
                act_func = Activation.map_name(layer.f)
                file_name_w += f'{layer_name}_{act_func}.npy'
                file_name_b += f'{layer_name}_{act_func}.npy'
            elif isinstance(layer, Softmax):
                file_name_w += f'{layer_name}.npy'
                file_name_b += f'{layer_name}.npy'
            else:
                raise ValueError()

            np.save(direct_path / file_name_w, layer.w)
            np.save(direct_path / file_name_b, layer.b)

    def _cost(self, x, y) -> float:
        return self.loss.f(self.predict(x), y)

    def _accuracy(self, x, y) -> float:
        _y_pre = np.argmax(self.predict(x), axis=0)

        if y is self.y_train:
            _y_true = np.argmax(self.y_train, axis=0)
        else:
            _y_true = np.argmax(self._one_hot(y), axis=0)

        return np.sum(_y_pre == _y_true) / _y_true.size

    def _forward(self, x):
        for layer in self.sequential:
            x = layer.forward(x)
        return x

    def _train(self, start, end, alpha):
        pred = self._forward(self.x_train[:, start:end])
        C_gradient = self.loss.backward(pred, self.y_train[:, start:end])

        w = None

        for layer in reversed(self.sequential):
            C_gradient, w = layer.backward(C_gradient, w, alpha)

    @staticmethod
    def _one_hot(y_true: np.ndarray) -> np.ndarray:
        if y_true.ndim > 2:
            raise ValueError()

        y_true = y_true.flatten()
        _category = np.max(y_true) + 1
        _result = np.zeros((_category, y_true.shape[0]))
        _result[y_true, np.arange(y_true.size)] = 1
        return _result
