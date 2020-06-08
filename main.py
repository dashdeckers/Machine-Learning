import random
import string

import numpy as np
import torch
import torch.nn as nn
# from torchtext.datasets import IMDB as dataset
from torchtext.datasets import WikiText2 as dataset
from unidecode import unidecode


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


def iters2seqs(iters):
    seq_sets = []
    for iter in iters:
        words = list()
        [words.extend(item.text) for item in iter.data()]
        text = ''.join([unidecode(c) for c in words
                        if unidecode(c).isalpha() or unidecode(c) == ' '])
        text = text.lower()
        # Like a sliding window, get seq_length chars one char at a time
        sequences = [text[i: i + seq_length] for i in
                     range(len(text) - seq_length)]
        max_batches = int(len(sequences)/batch_size)
        sequences = sequences[:max_batches*batch_size]
        seq_sets.append(sequences)
    return seq_sets


def train_test(sequences, train=True, epoch=0, max_n_batches=-1):
    if train:
        batch_n = 0
        # Cut sequences into batches
        for batch in [train_sequences[i: i+batch_size] for i in
                      range(0, len(train_sequences), batch_size)]:
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

            guesses = torch.argmax(reconstructed, dim=1)
            real_answers = torch.argmax(batch_to_tensor(batch, as_input=False),
                                        dim=1)
            corrects = torch.eq(guesses, real_answers)
            acc = sum(corrects).item() / batch_size

            if(batch_n % 500 == 0):
                print('[%d, %d] loss: %.3f, acc: %.3f' %
                      (batch_n, epoch, loss.item(), acc))
            if max_n_batches == batch_n:
                return
    else:
        print(len(sequences))
        with torch.no_grad():
            n_batches = 0
            total_acc = 0
            for batch in [sequences[i: i+batch_size] for i in
                          range(0, len(sequences), batch_size)]:
                corrupted = corrupt_batch(batch)
                reconstructed = model(batch_to_tensor(corrupted))
                loss = criterion(reconstructed, torch.argmax(
                    batch_to_tensor(batch, as_input=False), dim=1))
                guesses = torch.argmax(reconstructed, dim=1)
                real_answers = torch.argmax(batch_to_tensor(
                            batch, as_input=False), dim=1)
                corrects = torch.eq(guesses, real_answers)
                total_acc += sum(corrects).item() / batch_size
                n_batches += 1
                if max_n_batches == n_batches:
                    break

            print('[Validation, %d] loss: %.3f, test_acc: %.3f' %
                  (epoch, loss.item(), total_acc/n_batches))


print('\n', '*' * 5, f'Defining the model', '*' * 5)

num_epochs = 3
batch_size = 2048
seq_length = 11
input_size = 27

# Taking e.g. 4 layers gives bad results
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
int2char = dict(enumerate(string.ascii_lowercase + ' '))
char2int = {ch: ii for ii, ch in int2char.items()}

iters = dataset.iters(1)
train_sequences, val_sequences, test_sequences = iters2seqs(iters)


print('\n', '*' * 5, f'Training the model', '*' * 5)

for epoch in range(num_epochs):
    print('\n', '*' * 5, f'Epoch {epoch}', '*' * 5)
    train_test(train_sequences, train=True, epoch=epoch)
    train_test(val_sequences, train=False, epoch=epoch)
