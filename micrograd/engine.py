"""atomic unit of micrograd. Basis for building neurons, layers and MLPs."""

import math


class Value:
    def __init__(self, data: float, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=[self, other], _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        self._backward = _backward

        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, _children=[self, other], _op="-")

        def _backward():
            self.grad += out.grad
            other.grad += -out.grad

        self._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, [self, other], "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        self._backward = _backward

        return out

    def __pow__(self, power):
        data = self.data ** power

        def _backward():
            self.grad += out.grad * (power * self.data ** (power - 1))

        self._backward = _backward

        return Value(data, None, f"**{power}")

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data ** -1, [self, other], "/")

        def _backward():
            self.grad += out.grad * (other.data ** -1)
            other.grad += out.grad * (-self.data * other.data ** -2)

        self._backward = _backward

        return out

    def tanh(self,):
        out = Value(math.tanh(self.data), [self], "tanh")

        def _backward():
            self.grad = out.grad * (1 - out ** 2)

        self._backward = _backward

        return out

    def relu(self,):
        out = Value(0 if self.data < 0 else self.data, [self], "relu")

        def _backward():
            self.grad += out.grad * (out.data > 0)

        self._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -1 * self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self ** -1 * other

    def __repr__(self):
        msg = f"Value(data = {str(self.data)}, grad = {str(self.grad)})"
        return msg

    def backward(self,):
        topo = list()
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
