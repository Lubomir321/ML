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

def main():
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)

if __name__ == "__main__":
    main()