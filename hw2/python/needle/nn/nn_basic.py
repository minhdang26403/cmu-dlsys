"""The module."""

from typing import Any

import needle.init as init
from needle import ops
from needle.autograd import Tensor


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )

        self.bias = (
            Parameter(
                init.kaiming_uniform(
                    out_features, 1, device=device, dtype=dtype
                ).transpose()
            )
            if bias
            else None
        )
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias:
            out += ops.broadcast_to(self.bias, out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        flattened_dim = 1
        for dim in X.shape[1:]:
            flattened_dim *= dim
        return ops.reshape(X, (X.shape[0], flattened_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        # Create a one-hot encoding of the labels
        one_hot_y = init.one_hot(num_classes, y)
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        loss_per_example = log_sum_exp - ops.summation(logits * one_hot_y, axes=(1,))
        return ops.summation(loss_per_example) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            # At test time, we use the running mean and variance to normalize the input.
            return ops.broadcast_to(self.weight, x.shape) * (
                x - ops.broadcast_to(self.running_mean, x.shape)
            ) / (
                ops.broadcast_to(self.running_var, x.shape) + self.eps
            ) ** 0.5 + ops.broadcast_to(self.bias, x.shape)

        batch_size = x.shape[0]
        # Compute the mean and variance of the batch.
        mean = ops.summation(x, axes=0) / batch_size
        diff = x - ops.broadcast_to(mean, x.shape)
        var = ops.summation(diff**2, axes=0) / batch_size

        # Update the running mean and variance to be used at test time.
        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        # Normalize the input and apply the weight and bias.
        return ops.broadcast_to(self.weight, x.shape) * (
            x - ops.broadcast_to(mean, x.shape)
        ) / (ops.broadcast_to(var, x.shape) + self.eps) ** 0.5 + ops.broadcast_to(
            self.bias, x.shape
        )
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION

        # The weight of a feature is the same for each example in a mini-batch.
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        # The bias of a feature is the same for each example in a mini-batch.
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        full_shape = x.shape
        reduced_shape = (x.shape[0], 1)
        # Since our summation API does not have keepdims=True, we need to reshape the
        # result to add a dimension to simulate the keepdims=True behavior. This is
        # necessary for broadcasting to work correctly.
        mean = ops.reshape(ops.summation(x, axes=1) / self.dim, reduced_shape)
        diff = x - ops.broadcast_to(mean, full_shape)
        var = ops.reshape(ops.summation(diff**2, axes=1) / self.dim, reduced_shape)
        normalized_x = diff / (ops.broadcast_to(var, full_shape) + self.eps) ** 0.5

        return ops.broadcast_to(self.weight, x.shape) * normalized_x + ops.broadcast_to(
            self.bias, x.shape
        )
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
