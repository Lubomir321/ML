class Value:
    def __init__(self, data):
        self.data = float(data)
        self._prev = set()

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data)
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        out._prev = {self, other}
        return out

def main():
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._prev)

if __name__ == "__main__":
    main()