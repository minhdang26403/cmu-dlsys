"""Optimization module"""

from collections import defaultdict


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, p in enumerate(self.params):
            # Apply weight decay to the gradient.
            # Note: This operation should not be tracked for gradient computation
            # so operate on the underlying data arrays.
            grad = p.grad.data + self.weight_decay * p.data

            # Get the old velocity.
            v_old = self.u[i]

            # Compute the new velocity using the momentum rule
            v_new = self.momentum * v_old + (1 - self.momentum) * grad

            # Update the parameter using the new velocity.
            # This must be done in a way that doesn't build up the computation graph,
            # so we operate on the raw data arrays.
            p.data = p.data - self.lr * v_new.data

            # Save the new velocity for the next iteration.
            self.u[i] = v_new
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p.grad.data + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data = p.data - self.lr * m_hat / (v_hat**0.5 + self.eps)
        ### END YOUR SOLUTION
