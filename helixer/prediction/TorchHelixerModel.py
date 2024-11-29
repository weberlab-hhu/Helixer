import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from helixer.core import strs
from HelixerModel import HelixerModel

class CustomMockDataset(Dataset):
    def __init__(self, shape):
        self.shape = shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = torch.rand(self.shape[1:])
        y_idx = torch.max(x, dim=-1).values * 4 // 1
        y_idx = y_idx.to(torch.int64)
        return x, y_idx

class HybridSequence:
    #SequenceCls(model=self, h5_files=self.h5_trains, mode='train', batch_size=self.batch_size,
    #                       shuffle=True)
    LEN = 200
    CLASSES = 4


    # Create data loaders.
    def __init__(self, model, h5_files, mode, batch_size, shuffle) -> None:

        self.batch_size = batch_size
        self.model = model
        if mode == strs.TRAIN:
            n_examples = 12800
        else:
            n_examples = 128

        self.data = CustomMockDataset((n_examples, self.LEN, self.CLASSES))

        self.loader = DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)


class HybridModel(HelixerModel):
    def __init__(self, cli_args=None):
        super().__init__(cli_args)
        self.parse_args()
    
    def sequence_cls(self):
        return HybridSequence 
    
    def setup_model(self):
        return NeuralNetwork(HybridSequence.CLASSES).to(self.device)

    def compile_model(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)

   


#for X, y in test_dataloader:
#    print(f"Shape of X: {X.shape}")
#    print(f"Shape of y: {y.shape} {y.dtype}")
#    break


# Get cpu, gpu or mps device for training.

#print(f"Using {device} device")


class Transpose_1_2(nn.Module):
    def __init__(self):
        super(Transpose_1_2, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 2)  # swap class * len dimensions, not batch

class bLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = nn.LSTM(input_size=self.input_size,
                             hidden_size=self.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):

        x, _ = self.layer(x)
        return x


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            Transpose_1_2(),
            nn.Conv1d(n_classes, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            Transpose_1_2(),
            #nn.Conv1d(32, CLASSES, kernel_size=3, padding='same'),
            bLSTM(32, int(n_classes // 2)),
            nn.Linear(34, n_classes)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits






def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    # mini hard-code for testing:
    device = "gpu"
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # expected to be a tuple, so why the complaining, PyCharm?
    test_loss /= num_batches
    correct /= size * y.shape[1]
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    model = HybridModel()
    model.run()
#epochs = 3
#for t in range(epochs):
#    print(f"Epoch {t+1}\n-------------------------------")
#    train(train_dataloader, model, loss_fn, optimizer)
#    test(test_dataloader, model, loss_fn)


#torch.save(model.state_dict(), "model.pth")
#
#
#model = NeuralNetwork().to(device)
#model.load_state_dict(torch.load("model.pth"))
#
#
#model.eval()
#for x, y in test_dataloader:
#    break
#
#with torch.no_grad():
#    x = x.to(device)
#    #x = nn.Flatten()(x)
#    pred = model(x)
#    predicted, actual = pred[0].argmax(0), y[0]
#    print(f'Predicted: _, Actual: _ \n------------')
#    for y_hat_i, y_i in zip(predicted, actual):
#        print(y_hat_i.item(), y_i.item())


# goal 1 - go from something shaped [batch, 20k, 4] to something shaped [batch, 20k, 4]
# goal 2 - do the above with CNN + LSTM
# goal 3 (which will need breaking into subgoals)- connect the above to be called via HelixerModel.py

# todo: test Network (imitation of HybridModel in torch)
# todo: is this class a little overcomplicated?
class Network(HelixerModel):
    def __init__(self, cnn_layers, filter_depth, kernel_size, pool_size, lstm_layers,
                 units, n_classes, dropout1, dropout2):
        super().__init__()
        self.cnn_layers = cnn_layers
        self.filter_depth = filter_depth
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lstm_layers = lstm_layers
        self.units = units
        self.n_classes = n_classes
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        # needs to be at the end
        self.hparams = self.get_hparams()

        # WARNING: without RNA-seq coverage support so far

        # Add CNN stack
        # ---------------------------------
        self.cnn_blstm_stack = nn.Sequential()

        self.cnn_blstm_stack.append(self.TransposeDims())
        self.cnn_blstm_stack.append(nn.Conv1d(n_classes, self.filter_depth, self.kernel_size, padding='same'))
        self.cnn_blstm_stack.append(nn.ReLU())

        # if there are additional CNN layers
        for _ in range(self.cnn_layers - 1):
            # will NOT work like tensorflow, because of diff. available parameters and diff. definition of momentum
            self.cnn_blstm_stack.append(nn.BatchNorm1d(self.filter_depth))
            self.cnn_blstm_stack.append(nn.Conv1d(n_classes, self.filter_depth, self.kernel_size, padding='same'))
            self.cnn_blstm_stack.append(nn.ReLU())

        self.cnn_blstm_stack.append(self.TransposeDims())

        # Add bLSTM (and others) stack
        # --------------------------------
        if self.pool_size > 1:
            self.cnn_blstm_stack.append(self.Reshape((-1, self.pool_size * self.filter_depth)))

        if self.dropout1 > 0.0:
            self.cnn_blstm_stack.append(nn.Dropout(self.dropout1))

        self.cnn_blstm_stack.append(self.bLSTM(self.filter_depth, self.units, self.lstm_layers))

        # do not use recurrent dropout, but dropout on the output of the LSTM stack
        if self.dropout2 > 0.0:
            self.cnn_blstm_stack.append(nn.Dropout(self.dropout2))

        self.cnn_blstm_stack.append(self.ModelHat(self.units, self.pool_size, self.predict_phase))

    class TransposeDims(nn.Module):
        def __init__(self):
            super(Network.TransposeDims, self).__init__()

        @staticmethod
        def forward(x):
            return torch.transpose(x, 1, 2)  # swap class * len dimensions, not batch

    class bLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, lstm_layers, *args, **kwargs):
            super(Network.bLSTM, self).__init__()
            super().__init__(*args, **kwargs)
            self.layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=lstm_layers, batch_first=True, bidirectional=True)

        def forward(self, x):
            x, _ = self.layer(x)
            return x

    class Reshape(nn.Module):
        def __init__(self, new_shape, *args, **kwargs):
            super(Network.Reshape, self).__init__()
            super().__init__(*args, **kwargs)
            self.new_shape = new_shape

        def forward(self, x):
            return x.view(new_shape=self.new_shape)

    class ModelHat(nn.Module):
        def __init__(self, units, pool_size, predict_phase):
            super(Network.ModelHat, self).__init__()
            self.predict_phase = predict_phase
            if self.predict_phase:
                # right input size?, bidirectional lstm should double the given units
                self.linear_layer = nn.Linear(units * 2, pool_size * 4 * 2)

            else:
                self.linear_layer = nn.Linear(units * 2, pool_size * 4)

            self.output_stack = nn.Sequential()
            self.output_stack.append(Network.Reshape((-1, pool_size * 4)))
            self.output_stack.append(nn.Softmax(dim=2))  # last dim

        def forward(self, x):
            if self.predict_phase:
                x = self.linear_layer(x)
                x_genic, x_phase = torch.tensor_split(x, 2, dim=-1)
                x_genic = self.output_stack(x_genic)
                x_phase = self.output_stack(x_phase)
                return [x_genic, x_phase]

            else:
                x = self.linear_layer(x)
                x = self.output_stack(x)
                return [x]


    def forward(self, x):
        logits = self.cnn_blstm_stack(x)
        return logits
