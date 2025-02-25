from task01 import Value
from task07 import draw_dot
def main() -> None:
    # Define perceptron inputs and weights
    x1 = Value(2.0)  # Input 1
    x2 = Value(-3.0)  # Input 2
    w1 = Value(-0.5)  # Weight 1
    w2 = Value(1.0)  # Weight 2
    b = Value(6.8813735870195432)  # Bias

    # Perceptron forward pass: logit = x1*w1 + x2*w2 + b
    logit = x1 * w1 + x2 * w2 + b

    # Apply the tanh activation function
    output = logit.tanh()

    # Backward pass
    output.backward()

    # Visualize the computational graph
    dot = draw_dot(output)
    dot.render(directory='./graphviz_output', filename='tanh_perceptron', view=True)

    # Print results
    print(f"Logit value (before activation): {logit.data}")
    print(f"Output value (after tanh activation): {output.data}")
    print(f"Gradients: x1.grad={x1.grad}, x2.grad={x2.grad}, w1.grad={w1.grad}, w2.grad={w2.grad}, b.grad={b.grad}")

if __name__ == "__main__":
    main()