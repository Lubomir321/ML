class Value:
    def __init__(self, data: float, prev=None, op=None):
        self.data = data
        self.grad = 0.0  # Gradient initialized to 0
        self._prev = prev if prev else set()
        self._op = op
        self._backward = lambda: None  # Default backward method

    def __str__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: 'Value') -> 'Value':
        result = Value(self.data + other.data, prev={self, other}, op='+')

        def _backward():
            # Accumulate gradients instead of overwriting them
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward
        return result

    def __mul__(self, other: 'Value') -> 'Value':
        result = Value(self.data * other.data, prev={self, other}, op='*')

        def _backward():
            # Accumulate gradients instead of overwriting them
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = _backward
        return result

    def tanh(self) -> 'Value':
        import math
        result = Value(math.tanh(self.data), prev={self}, op='tanh')

        def _backward():
            # Accumulate gradients instead of overwriting them
            self.grad += (1 - result.data**2) * result.grad

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


# Test Case for Task 17
def main():
    # Define variables
    x = Value(1.0)
    y = x + x

    # Backward pass
    y.backward()

    # Print results
    print(f"x.data = {x.data}, x.grad = {x.grad}")
    print(f"y.data = {y.data}, y.grad = {y.grad}")


if __name__ == "__main__":
    main()
