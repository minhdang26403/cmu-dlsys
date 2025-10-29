from typing import Optional

import numpy as array_api

from ..autograd import NDArray, Tensor, TensorOp
from .ops_mathematic import broadcast_to, exp, reshape, summation


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:  # type: ignore[override]
        ### BEGIN YOUR SOLUTION
        # Need to reshape the logsumexp to shape that allows broadcasting.
        reduced_shape = list(Z.shape)
        reduced_shape[1] = 1

        # We can compute softmax via logsumexp and reuse the code.
        log_sum_exp = array_api.reshape(LogSumExp(axes=(1,)).compute(Z), reduced_shape)

        # Explicit broadcast to make the operation work with generic ndarray backend.
        return Z - array_api.broadcast_to(log_sum_exp, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):  # type: ignore[override]
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        assert isinstance(Z, Tensor)
        # Get the softmax of the input.
        softmax_Z = exp(logsoftmax(Z))
        assert Z.shape == softmax_Z.shape

        # Need to reshape the out_grad to shape that allows broadcasting.
        reduced_shape = list(Z.shape)
        reduced_shape[1] = 1
        sum_out_grad = reshape(summation(out_grad, axes=(1,)), reduced_shape)

        return out_grad - softmax_Z * broadcast_to(sum_out_grad, Z.shape)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:  # type: ignore[override]
        ### BEGIN YOUR SOLUTION
        # Preserve the shape of the input to perform element-wise subtraction
        self.max_Z = array_api.max(Z, axis=self.axes, keepdims=True)

        # Compute the log-sum-exp
        log_sum_exp = array_api.log(
            array_api.sum(array_api.exp(Z - self.max_Z), axis=self.axes)
        )

        # Need to squeeze the axes to remove the broadcasted dimension after summation.
        return log_sum_exp + self.max_Z.squeeze(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):  # type: ignore[override]
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        assert isinstance(Z, Tensor)
        # We don't have max API for Tensor, so we need to use array_api to get the max.
        max_Z = array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True)
        exp_Z = exp(Z - max_Z)

        # During the summation, the axes are summed, thus the summed axes are removed.
        # Therefore, we need to construct the shape that allows broadcasting to the
        # original shape.
        reduced_shape = list(Z.shape)
        if self.axes:
            for axies in self.axes:
                reduced_shape[axies] = 1
        else:
            reduced_shape = [1] * len(Z.shape)
        # Retain the shape after the summation. This has the same effect as the
        # keepdims=True parameter in ndarray.sum() of numpy.
        exp_sum = reshape(summation(exp_Z, axes=self.axes), reduced_shape)

        # out_grad's axes are removed, thus we need to reshape it to the original shape.
        return (
            broadcast_to(reshape(out_grad, reduced_shape), Z.shape)
            * exp_Z
            / broadcast_to(exp_sum, Z.shape)
        )
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
