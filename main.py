import random
import string

import numpy as np
import torch
import torch.nn as nn
import unidecode
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
        lstm_output, _ = self.lstm(
            batch.view(self.batch_size, self.seq_length, self.input_size)
            .float()#,
            #self.hidden
        )
        prediction = self.linear(
            lstm_output.contiguous()
            .view(self.batch_size, self.seq_length * self.hidden_size * 2,)
        )
        return self.softmax(prediction)#.argmax()


def corrupt_batch(batch, chance=0.1):
    # just sets some char to 'z' right now
    def corrupt(sequence, chance=chance):
        if np.random.random() < chance:
            mid = int(np.ceil(len(sequence) / 2))
            return (sequence[:mid-1]
                    + random.sample(string.ascii_lowercase, 1)[0]
                    + sequence[mid:])
        else:
            return sequence

    return list(map(corrupt, batch))


def one_hot_code(input):
    # Toggle input between one-hot encoded version and string version
    if(isinstance(input, str)):
        # Numpy one-hot
        encoded = np.array([char2int[char] for char in input])
        one_hot = np.zeros((len(encoded), len(int2char)))
        one_hot[np.arange(len(encoded)), encoded] = 1
        return one_hot
    else:
        # Take max value and translate it into char
        values = np.argmax(input, axis=1)
        decoded = [int2char[i] for i in values]
        return "".join(decoded)


def batch_to_tensor(batch, as_input=True):
    if as_input:
        ohc = [one_hot_code(entry) for entry in batch]
    else:
        mid = int(np.ceil(seq_length / 2))
        ohc = [one_hot_code(entry[mid]) for entry in batch]
    return torch.as_tensor(np.vstack(ohc))


def tensor_to_batch(tensor):
    return one_hot_code(tensor.numpy())


print('\n', '*' * 5, f'Defining the model', '*' * 5)

num_epochs = 1
batch_size = 2048
seq_length = 11
input_size = 27

num_layers = 2
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
text = ''.join([c for c in words if c.isalpha() or c == ' '])
text = unidecode.unidecode(text.lower())

int2char = dict(enumerate(string.ascii_lowercase + ' '))
char2int = {ch: ii for ii, ch in int2char.items()}


print('\n', '*' * 5, f'Training the model', '*' * 5)

# Like a sliding window, get seq_length chars one char at a time
sequences = [text[i: i + seq_length] for i in
             range(len(text) - seq_length)]
print(len(sequences))
for sequence in sequences:
    if len(sequence) != seq_length:
        print(len(sequence))
        print(sequence)

sequences = sequences[:10000*batch_size]
for epoch in range(num_epochs):
    print('\n', '*' * 5, f'Epoch {epoch}', '*' * 5)

    batch_n=0
    # Cut sequences into batches
    for batch in [sequences[i: i+batch_size] for i in
                  range(0, len(sequences), batch_size)]:
        batch_n += 1
        # Corrupt a batch
        corrupted = corrupt_batch(batch)
        # Try reconstructing it
        reconstructed = model(batch_to_tensor(corrupted))

        # Compare
        loss = criterion(reconstructed, torch.argmax(
            batch_to_tensor(batch, as_input=False), dim=1))

        # Backprop
        loss.backward(retain_graph=True)
        optimizer.step()
        model.zero_grad()

        guesses = torch.eq(torch.argmax(reconstructed, dim=1))
        real_answers = torch.argmax(batch_to_tensor(batch, as_input=False),
                                    dim=1)
        corrects = sum(guesses, real_answers).item()
        acc = corrects / batch_size

        if(batch_n % 10 == 0):
            print("Batch " + str(batch_n))
            print('[%d, %d] loss: %.3f, test_acc: %.3f' %
                  (batch_n, epoch, loss.item(), acc))
