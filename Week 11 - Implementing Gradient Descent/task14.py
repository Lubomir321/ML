from task01 import Value
from task07 import draw_dot
import math

def main() -> None:
    # Define perceptron inputs and weights
    x1 = Value(2.0)  # Input 1
    x2 = Value(0)  # Input 2
    w1 = Value(-3)  # Weight 1
    w2 = Value(1.0)  # Weight 2
    b = Value(6.8813735870195432)  # Bias

    # Forward pass: compute logit and output
    logit = x1.data * w1.data + x2.data * w2.data + b.data
    output = math.tanh(logit)

    # Manual backward pass
    d_output = 1.0  # Gradient of the output w.r.t itself
    d_logit = d_output * (1 - output**2)  # Gradient of the tanh

    # Gradients of inputs and weights
    x1_grad = d_logit * w1.data
    x2_grad = d_logit * w2.data
    w1_grad = d_logit * x1.data
    w2_grad = d_logit * x2.data
    b_grad = d_logit * 1  # Bias gradient is d_logit * 1

    # Print results
    print(f"Logit value (before activation): {logit}")
    print(f"Output value (after tanh activation): {output}")
    print(f"Manual Gradients: x1.grad={x1_grad}, x2.grad={x2_grad}, w1.grad={w1_grad}, w2.grad={w2_grad}, b.grad={b_grad}")

    # Visualize the computational graph
    x1.grad = x1_grad
    x2.grad = x2_grad
    w1.grad = w1_grad
    w2.grad = w2_grad
    b.grad = b_grad

    value_logit = Value(logit)
    value_output = value_logit.tanh()
    value_output.backward()

    dot = draw_dot(value_output)
    dot.render(directory='./graphviz_output', filename='manual_tanh_perceptron', view=True)

if __name__ == "__main__":
    main()
