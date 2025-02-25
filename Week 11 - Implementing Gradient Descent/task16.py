   def tanh(self) -> 'Value':
        result = Value(math.tanh(self.data), prev={self}, op='tanh')

        def _backward():
            self.grad += (1 - result.data**2) * result.grad

        result._backward = _backward
        return result