from graphviz import Digraph
class Value:
    def __init__(self, data):
        self.data = float(data)
        self._prev = set()
        self._op = ""
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

def manual_der(result):
    result.grad = 1.0

    def backpropagate(v):
        if v._op == "+":
            for child in v._prev:
                child.grad += 1 * v.grad
                backpropagate(child)
        elif v._op == "*":
            for child in v._prev:
                other = (v._prev - {child}).pop()
                child.grad += other.data * v.grad
                backpropagate(child)

    backpropagate(result)

def main():
    # Initial values
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    t = Value(-2.0)
    
    result = (x * y + z) * t

    print(f"Old L = {result.data}")

    manual_der(result)

    val = 0.01
    for i in [x,y,z,t,result]:
        i.data += val
        
    result = (x * y + z) * t
    print(f"New L = {result.data}")
    draw_dot(result).render(directory='./graphviz_output', view=True)

if __name__ == "__main__":
    main()