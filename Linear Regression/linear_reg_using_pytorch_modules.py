import torch
import torch.nn as nn

X = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        o = self.linear(x)
        return o

model = LinearRegression()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X)
    loss = criterion(y_pred, y)

    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch} :: Loss: {loss.item()}')


new_var = torch.tensor([[4.0]])
pred = model(new_var)
print(f'4 : {pred.item()}')