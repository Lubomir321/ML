from task01 import Value

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