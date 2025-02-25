class Value:
    def __init__(self, data: float, prev=None, op=None):
        self.data = data
        self.grad = 0.0  # Gradient initialized to 0
        self._prev = prev if prev else set()
        self._op = op

    def __str__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: 'Value') -> 'Value':
        result = Value(self.data + other.data, prev={self, other}, op='+')

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward
        return result

    def __mul__(self, other: 'Value') -> 'Value':
        result = Value(self.data * other.data, prev={self, other}, op='*')

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

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

# Function to trace nodes and edges
def trace(value: 'Value'):
    nodes, edges = set(), set()

    def build(value):
        if value not in nodes:
            nodes.add(value)
            for prev in value._prev:
                edges.add((prev, value))
                build(prev)

    build(value)
    return nodes, edges

def draw_dot(root: Value):
    import graphviz

    dot = graphviz.Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f"{{ data: {n.data} | grad: {n.grad} }}", shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def main() -> None:
    # Define perceptron inputs and weights
    x1 = Value(2.0)  # Input 1
    x2 = Value(-3.0)  # Input 2
    w1 = Value(-0.5)  # Weight 1
    w2 = Value(1.0)  # Weight 2
    b = Value(0.1)  # Bias

    # Perceptron forward pass: logit = x1*w1 + x2*w2 + b
    logit = x1 * w1 + x2 * w2 + b

    # Visualize the computation graph
    dot = draw_dot(logit)
    dot.render(directory='./graphviz_output', filename='perceptron', view=True)

    # Output the logit
    print(f"Logit value (without activation): {logit.data}")

if __name__ == "__main__":
    main()
