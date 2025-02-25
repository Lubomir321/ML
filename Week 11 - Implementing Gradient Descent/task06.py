class Value:
    def __init__(self, data):
        self.data = float(data)
        self._prev = set()
        self._op = ""

    def __repr__(self):
        return f"Value(data={self.data})"

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

    def build(v): #I know it is not good practice, but I did not remeber how to pass it by & at the moment of making
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(value)
    return nodes, edges

def main():
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z

    nodes, edges = trace(result)
    print(f'{nodes=}')
    print(f'{edges=}')

if __name__ == "__main__":
    main()
