import torch
import torch.nn as nn

X = torch.tensor(list([float(i)] for i in range(1, 7)))
y = torch.tensor(list([float(i)] for i in [0,0,0,1,1,1]))

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.001

epochs = 200
for i in range(epochs + 1):
    z = w * X + b
    pred = torch.sigmoid(z)

    loss = -(y * torch.log(pred) + (1-y) * torch.log(1 - pred)).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if i % 20 == 0:
        print(f'Epoch {i} :: loss {loss.item()}')