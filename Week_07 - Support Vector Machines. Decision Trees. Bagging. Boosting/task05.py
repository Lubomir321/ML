import numpy as np
import matplotlib.pyplot as plt

def log_loss(raw_model_output):
    return np.log(1 + np.exp(-raw_model_output)) #add link to formula

def hinge_loss(raw_model_output):
    return np.maximum(0,1-raw_model_output)

def main():
    raw_model_output=np.linspace(-2,2,1000)

    log_loss_values = log_loss(raw_model_output)
    hinge_loss_values = hinge_loss(raw_model_output)

    plt.plot(raw_model_output, log_loss_values, label='logistic', color='blue')
    plt.plot(raw_model_output, hinge_loss_values, label='hinge', color='orange')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()