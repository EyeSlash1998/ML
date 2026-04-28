import torch
import torch.nn as nn

X = torch.tensor(list([float(i)] for i in range(1, 7)))
y = torch.tensor(list([float(i)] for i in [0,0,0,1,1,1]))

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LogisticRegression()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 200
for epoch in range(epochs + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch} :: loss {loss.item()}')

model.eval()
with torch.no_grad():
    logits = model(X)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    accuracy = (preds == y).float().mean()

print(f'Accuracy: {accuracy}%')