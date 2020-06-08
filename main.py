import random
import string

import numpy as np
import torch
import torch.nn as nn
from torchtext.datasets import IMDB as dataset


# The model
class CharDenoiser(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, seq_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_size = input_size

        # LSTM:
        # Input shape = (batch, seq_len, inp_size)
        # Output shape = (batch, seq_len, num_directions * hidden_size)
        # Hidden shape = (num_layers * num_directions, batch, hidden_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(
            # x2 because bidirectional, same for hidden layer
            in_features=seq_length * hidden_size * 2,
            out_features=input_size
        )
        self.hidden = (
            torch.zeros(num_layers * 2, batch_size, hidden_size),
            torch.zeros(num_layers * 2, batch_size, hidden_size)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        lstm_output, self.hidden = self.lstm(
            batch.view(self.batch_size, self.seq_length, self.input_size),
            self.hidden
        )
        prediction = self.linear(
            lstm_output.contiguous().view(self.batch_size, self.input_size)
        )
        return self.softmax(prediction).argmax()


def corrupt_batch(batch, chance=0.1):
    # just sets some char to 'z' right now
    def corrupt(sequence, chance=chance):
        if np.random.random() < chance:
            mid = int(np.ceil(len(sequence) / 2))

            return (sequence[:mid-1]
                    + random.sample(string.ascii_lowercase, 1)[0]
                    + sequence[mid+1:])
        else:
            return sequence

    return list(map(corrupt, batch))


def one_hot_code(input):
    if(isinstance(input, str)):
        # Numpy one-hot
        encoded = np.array([char2int[char] for char in input])
        one_hot = np.zeros((len(encoded), len(int2char)))
        one_hot[np.arange(len(encoded)), encoded] = 1
        return one_hot
    else:
        values = np.argmax(input, axis=1)
        decoded = [int2char[i] for i in values]
        return "".join(decoded)


def batch_to_tensor(batch):
    pass


def tensor_to_batch(tensor):
    pass


print('\n', '*' * 5, f'Defining the model', '*' * 5)

num_epochs = 1
batch_size = 32
seq_length = 11
input_size = 128

num_layers = 1
hidden_size = 10

model = CharDenoiser(
    input_size,
    hidden_size,
    num_layers,
    batch_size,
    seq_length
)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print('Model:', model)


print('\n', '*' * 5, f'Getting the data', '*' * 5)

# Just get a long af text as toy data
train_iter, test_iter = dataset.iters(1)
words = list()
[words.extend(item.text) for item in train_iter.data()]
text = ' '.join(words)
# preprocess: filter(c.isalpha() or c == ' ')
# preprocess: lowercase

int2char = dict(enumerate(string.ascii_lowercase + ' '))
char2int = {ch: ii for ii, ch in int2char.items()}


print('\n', '*' * 5, f'Training the model', '*' * 5)

# Like a sliding window, get seq_length chars one char at a time
sequences = [text[i: i + seq_length] for i in
             range(len(text) - seq_length)]

for epoch in range(num_epochs):
    print('\n', '*' * 5, f'Epoch {epoch}', '*' * 5)

    # Cut sequences into batches
    for batch in [sequences[i: i+batch_size] for i in
                  range(0, len(sequences), batch_size)]:

        # Corrupt a batch
        corrupted = corrupt_batch(batch)
        # Try reconstructing it
        reconstructed = model(batch_to_tensor(corrupted))

        # Compare
        loss = criterion(tensor_to_batch(reconstructed), batch)

        # Backprop
        loss.backward()
        optimizer.step()
        model.zero_grad()
