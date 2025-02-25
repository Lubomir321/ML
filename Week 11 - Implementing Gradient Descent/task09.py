from graphviz import Digraph

class Value:
    def __init__(self, data, label=""):
        self.data = float(data)
        self._prev = set()
        self._op = ""
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data)
        out._prev = {self, other}
        out._op = "+"
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        out._prev = {self, other}
        out._op = "*"
        return out

def trace(value):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(value)
    return nodes, edges

def draw_dot(root: Value) -> Digraph:
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    labels = list("abcdefghijklmnopqrstuvwxyz")

    for i, n in enumerate(nodes):
        uid = str(id(n))
        label = f"{{ {labels[i]} | data: {n.data} | grad: {n.grad} }}"
        dot.node(name=uid, label=label, shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def main():
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    t = Value(-2.0)
    result = (x * y + z) * t

    draw_dot(result).render(directory='./graphviz_output', view=True)

if __name__ == "__main__":
    main()
