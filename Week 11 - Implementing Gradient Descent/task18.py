class Value:
    def __init__(self, data: float, prev=None, op=None):
        self.data = data
        self.grad = 0.0  # Gradient initialized to 0
        self._prev = prev if prev else set()
        self._op = op
        self._backward = lambda: None  # Default backward method

    def __str__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    # Adding a float to a Value object
    def __add__(self, other):
        if isinstance(other, Value):
            result = Value(self.data + other.data, prev={self, other}, op='+')
        elif isinstance(other, (int, float)):
            result = Value(self.data + other, prev={self}, op='+')

        def _backward():
            self.grad += result.grad
            if isinstance(other, Value):
                other.grad += result.grad

        result._backward = _backward
        return result

    # Support reverse add: float + Value
    __radd__ = __add__

    # Multiplying a Value object with a float
    def __mul__(self, other):
        if isinstance(other, Value):
            result = Value(self.data * other.data, prev={self, other}, op='*')
        elif isinstance(other, (int, float)):
            result = Value(self.data * other, prev={self}, op='*')

        def _backward():
            self.grad += (other if isinstance(other, (int, float)) else other.data) * result.grad
            if isinstance(other, Value):
                other.grad += self.data * result.grad

        result._backward = _backward
        return result

    # Support reverse multiplication: float * Value
    __rmul__ = __mul__

    # Dividing a Value object by a float
    def __truediv__(self, other):
        if isinstance(other, Value):
            raise NotImplementedError("Division by another Value is not supported yet.")
        elif isinstance(other, (int, float)):
            result = Value(self.data / other, prev={self}, op='/')

        def _backward():
            self.grad += (1 / other) * result.grad

        result._backward = _backward
        return result

    # Exponentiation of a Value object with a float
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Only exponentiation with floats is supported.")
        result = Value(self.data**other, prev={self}, op=f'**{other}')

        def _backward():
            self.grad += other * (self.data**(other - 1)) * result.grad

        result._backward = _backward
        return result

    # Exponentiation of Euler's number with a Value object
    def exp(self):
        import math
        result = Value(math.exp(self.data), prev={self}, op='exp')

        def _backward():
            self.grad += result.data * result.grad

        result._backward = _backward
        return result

    def backward(self):
        # Initialize gradient of the output node to 1
        self.grad = 1.0

        # Perform a topological sort of the computation graph to process dependencies
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Apply the chain rule in reverse topological order
        for node in reversed(topo):
            if hasattr(node, '_backward'):
                node._backward()

def main() -> None:
    x = Value(2.0, label='x')

    expected = Value(4.0)

    actuals = {
        'actual_sum_l': x + 2.0,
        'actual_sum_r': 2.0 + x,
        'actual_mul_l': x * 2.0,
        'actual_mul_r': 2.0 * x,
        'actual_div_r': (x + 6.0) / 2.0,
        'actual_pow_l': x**2,
        'actual_exp_e': x**2,
    }

    assert x.exp().data == np.exp(
        2), f"Mismatch for exponentiating Euler's number: expected {np.exp(2)}, but got {x.exp().data}."

    for actual_name, actual_value in actuals.items():
        assert actual_value.data == expected.data, f'Mismatch for {actual_name}: expected {expected.data}, but got {actual_value.data}.'

    print('All tests passed!')

if __name__ == "__main__":
    main()