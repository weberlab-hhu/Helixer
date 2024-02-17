import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class CustomMockDataset(Dataset):
    def __init__(self, shape):
        self.shape = shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = torch.rand(self.shape[1:])
        y_idx = torch.max(x, axis=-1).values * 4 // 1 
        y_idx = y_idx.to(torch.int64)
        return x, y_idx

LEN = 200
CLASSES = 4

training_data = CustomMockDataset((12800, LEN, CLASSES))
test_data = CustomMockDataset((100, LEN, CLASSES))


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LEN * CLASSES, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, LEN * CLASSES),
            nn.Unflatten(1, (CLASSES, LEN))
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size * y.shape[1]
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)


torch.save(model.state_dict(), "model.pth")


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))


model.eval()
for x, y in test_dataloader:
    break

with torch.no_grad():
    x = x.to(device)
    #x = nn.Flatten()(x)
    pred = model(x)
    predicted, actual = pred[0].argmax(0), y[0]
    print(f'Predicted: _, Actual: _ \n------------')
    for y_hat_i, y_i in zip(predicted, actual):
        print(y_hat_i.item(), y_i.item())


# goal 1 - go from something shaped [batch, 20k, 4] to something shaped [batch, 20k, 4]
# goal 2 - do the above with CNN + LSTM
# goal 3 (which will need breaking into subgoals)- connect the above to be called via HelixerModel.py