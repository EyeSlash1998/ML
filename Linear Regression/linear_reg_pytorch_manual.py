import torch
import numpy as np
import matplotlib.pyplot as plt


x = torch.arange(-5, 5, 0.1).view(-1, 1)

print(f'X Size: {x.shape}')

# Defining a fixed function
func = -5 * x

# Plotting to see result
plt.plot(x.numpy(), func.numpy(), 'r', label='func')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, color='y')
plt.show()


# Creating y with Gaussian noise
y = func + 0.4 * torch.randn(x.size())

# Plotting
plt.plot(x.numpy(), func.numpy(), 'r', label='func')
plt.plot(x.numpy(), y.numpy(), 'b+', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, color='y')
plt.show()

# Defining Functions
def forward(x):
    return w * x + b


def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)


# Hyperparameter Defining
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

step_size = 0.1
loss_list = []
iterations = 20

# Model Training
for i in range(iterations):
    y_pred = forward(x)
    loss = criterion(y_pred, y)
    loss_list.append(loss.item())

    # Computer Gradient
    loss.backward()

    # Weight Update
    with torch.no_grad():
        w -= step_size * w.grad
        b -= step_size * b.grad

    # Setting gradients to zero
    w.grad.data.zero_()
    b.grad.data.zero_()
    print(f'Iteration {i} :: Loss {loss.item()} Weight: {w.item()} Bias {b.item()}')

plt.plot(loss_list, 'r')
plt.tight_layout()
plt.grid(True, color='y')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
