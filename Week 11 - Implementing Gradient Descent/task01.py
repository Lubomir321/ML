class Value:
    def __init__(self, data):
        self.data = float(data)

    def __repr__(self):
        return f"Value(data={self.data})"

def main():
    value1 = Value(5)
    print(value1)

    value2 = Value(6)
    print(value2)

if __name__ == "__main__":
    main()