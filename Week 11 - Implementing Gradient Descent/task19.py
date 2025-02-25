class Value:
    def __init__(self, data: float, prev=None, op=None):
        self.data = data
        self.grad = 0.0  # Gradient initialized to 0
        self._prev = prev if prev else set()
        self._op = op
        self._backward = lambda: None  # Default backward method

    # Exponentiation of Euler's number with a Value object
    def exp(self):
        import math
        result = Value(math.exp(self.data), prev={self}, op='exp')

        def _backward():
            self.grad += result.data * result.grad

        result._backward = _backward
        return result

    # Hyperbolic tangent broken into components
    def tanh(self):
        # Forward pass: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        exp_pos = self.exp()  # e^x
        exp_neg = (-self).exp()  # e^(-x)
        numerator = exp_pos - exp_neg
        denominator = exp_pos + exp_neg
        result = numerator / denominator

        # Backward pass defined in terms of components
        def _backward():
            numerator.grad += (1 / denominator.data) * result.grad
            denominator.grad -= (numerator.data / (denominator.data**2)) * result.grad

        result._backward = _backward
        return result

    # Negation (needed for e^(-x))
    def __neg__(self):
        return self * -1.0

    # Addition, multiplication, etc., as defined in Task 18
    # [Refer to the code provided for Task 18]
