import itertools
import random
import string

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from torchtext.datasets import IMDB as dataset
from torchtext.datasets import WikiText2 as dataset
from unidecode import unidecode


# The model
class CharDenoiser(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 batch_size, seq_length, disjoint=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_size = input_size
        self.disjoint = disjoint

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

        if self.disjoint:
            # Could probably use something like copy() here
            self.classlstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Linear(
            in_features=seq_length * hidden_size * 2,
            out_features=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        lstm_output, _ = self.lstm(
            batch.view(self.batch_size, self.seq_length, self.input_size)
            .float()
        )
        prediction = self.linear(
            lstm_output.contiguous()
            .view(self.batch_size, self.seq_length * self.hidden_size * 2,)
        )
        if self.disjoint:
            classlstm_output, _ = self.classlstm(
                batch.view(self.batch_size, self.seq_length, self.input_size)
                .float()
            )
            classification = self.classifier(
                classlstm_output.contiguous()
                .view(self.batch_size, self.seq_length * self.hidden_size * 2,)
            )
        else:
            classification = self.classifier(
                lstm_output.contiguous()
                .view(self.batch_size, self.seq_length * self.hidden_size * 2,)
            )
        return torch.cat((self.softmax(prediction),
                          self.sigmoid(classification)), dim=1)


def corrupt_batch(batch, chance=0.3):
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


# Batch to tensor generates input and output data
# It's a bit of a misnomer as it does more stuff for output data
def batch_to_tensor(batch, as_input=True, alt=None):
    if as_input:
        ohc = [one_hot_code(entry) for entry in batch]
    else:
        assert(alt is not None)
        mid = int(np.ceil(seq_length / 2))-1
        ohc = []
        for i, entry in enumerate(batch):
            seq = one_hot_code(entry[mid])
            if entry[mid] == alt[i][mid]:
                corrupted = 0
            else:
                corrupted = 1
            seq = np.append(seq, corrupted)
            ohc.append(seq)
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


def determine_accuracy(output, batch, alt, batch_n=0, epoch=0, label="", logging=True):

    guesses = torch.argmax(output[:, :-1], dim=1)
    correct_output = batch_to_tensor(batch, as_input=False, alt=alt)
    real_answers = torch.argmax(correct_output[:, :-1], dim=1)

    pred_corrects = torch.eq(guesses, real_answers)
    class_corrects = torch.eq(output[:, -1].round(), correct_output[:, -1])
    # Either the classifier has to be correct
    # Or the classifier has to be false and the prediction correct
    corrects = class_corrects | (~(output[:, -1].round()
                                                .bool()) & pred_corrects)

    if logging:
        writer_dict.get('runs/Class_corrects/'+label).add_scalar(
                        'dat',
                        sum(class_corrects).item() / batch_size,
                        batch_n*(epoch+1))
        writer_dict.get('runs/Pred_corrects/'+label).add_scalar(
                        'dat',
                        sum(pred_corrects).item() / batch_size,
                        batch_n*(epoch+1))
        writer_dict.get('runs/True_corrects/'+label).add_scalar(
                        'dat',
                        sum(corrects).item() / batch_size,
                        batch_n*(epoch+1))

    Class_corrects = sum(class_corrects).item() / batch_size
    Pred_corrects = sum(pred_corrects).item() / batch_size
    True_corrects = sum(corrects).item() / batch_size
    return (Class_corrects, Pred_corrects, True_corrects)


def train_test(sequences, model, optimizer,
               train=True, epoch=0, max_n_batches=-1, label="", eval=False):
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
                batch_to_tensor(batch, as_input=False, alt=batch), dim=1))

            # Backprop
            loss.backward(retain_graph=True)
            optimizer.step()
            model.zero_grad()
            writer_dict.get('runs/loss/train/'+label).add_scalar(
                            'dat', loss.item(),
                            batch_n*(epoch+1))

            _, _, acc = determine_accuracy(reconstructed, batch, corrupted,
                                           batch_n, epoch, "train/" + label)
            if(batch_n % 500 == 0):
                print(label + '[%d, %d] loss: %.3f, acc: %.3f' %
                      (batch_n, epoch, loss.item(), acc))
            if max_n_batches == batch_n:
                return
    else:
        with torch.no_grad():
            n_batches = 0
            class_acc = 0
            pred_acc = 0
            true_acc = 0
            total_loss = 0
            for batch in [sequences[i: i+batch_size] for i in
                          range(0, len(sequences), batch_size)]:
                corrupted = corrupt_batch(batch)
                reconstructed = model(batch_to_tensor(corrupted))
                loss = criterion(reconstructed, torch.argmax(
                    batch_to_tensor(batch,
                                    as_input=False, alt=corrupted), dim=1))
                clacc, precc, truecc = determine_accuracy(
                                reconstructed, batch, corrupted, logging=False)
                total_loss += loss
                class_acc += clacc
                pred_acc += precc
                true_acc += truecc
                n_batches += 1
                if max_n_batches == n_batches:
                    break

            if not eval:
                test_type = "val"
            else:
                test_type = "test"

            writer_dict.get('runs/loss/' +
                            test_type + '/'+label).add_scalar(
                                'dat', total_loss/n_batches,
                                epoch*number_train_batch)
            writer_dict.get('runs/Class_corrects/' +
                            test_type + '/'+label).add_scalar(
                                'dat', class_acc/n_batches,
                                epoch*number_train_batch)
            writer_dict.get('runs/Pred_corrects/' +
                            test_type + '/'+label).add_scalar(
                                'dat', pred_acc/n_batches,
                                epoch*number_train_batch)
            writer_dict.get('runs/True_corrects/' +
                            test_type + '/'+label).add_scalar(
                                'dat', true_acc/n_batches,
                                epoch*number_train_batch)
            label = test_type + "/" + label
            print('[' + label + ' %d] loss: %.3f, test_acc: %.3f' %
                  (epoch, total_loss/n_batches, true_acc/n_batches))


print('\n', '*' * 5, f'Defining the model', '*' * 5)

torch.set_num_threads(12)
num_epochs = 2
batch_size = 2048
seq_length = 21
input_size = 27

# Taking e.g. 4 layers gives bad results
num_layers = 2
hidden_size = 32

model = CharDenoiser(
    input_size,
    hidden_size,
    num_layers,
    batch_size,
    seq_length,
    disjoint=False
)
disjoint_model = CharDenoiser(
    input_size,
    hidden_size,
    num_layers,
    batch_size,
    seq_length,
    disjoint=True
)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
disjoint_optimizer = torch.optim.Adam(disjoint_model.parameters())
print('Model:', model)

print('Disjoint Model:', model)


print('\n', '*' * 5, f'Getting the data', '*' * 5)

# Just get a long af text as toy data
int2char = dict(enumerate(string.ascii_lowercase + ' '))
char2int = {ch: ii for ii, ch in int2char.items()}

iters = dataset.iters(1)
train_sequences, val_sequences, test_sequences = iters2seqs(iters)

number_train_batch = len(train_sequences) / batch_size

runs = ["runs"]
metrics = ["loss", "True_corrects", "Pred_corrects", "Class_corrects"]
datas = ["train", "val", "test"]
joins = ["Conjoined", "Disjoined"]

combinations = list(itertools.product(runs, metrics, datas, joins))
writer_strings = ['/'.join(line)for line in combinations]

writers = [SummaryWriter(log_dir=string) for string in writer_strings]
writer_dict = dict(zip(writer_strings, writers))

print('\n', '*' * 5, f'Training the model', '*' * 5)

for epoch in range(num_epochs):
    print('\n', '*' * 5, f'Epoch {epoch}', '*' * 5)
    train_test(train_sequences, model=model,
               optimizer=optimizer, train=True,
               epoch=epoch, label="Conjoined")
    train_test(val_sequences, model=model,
               optimizer=optimizer, train=False,
               epoch=epoch, label="Conjoined")

    train_test(train_sequences, model=disjoint_model,
               optimizer=disjoint_optimizer, train=True,
               epoch=epoch, label="Disjoined")
    train_test(val_sequences, model=disjoint_model,
               optimizer=disjoint_optimizer, train=False,
               epoch=epoch, label="Disjoined")
train_test(test_sequences, model=model,
           optimizer=optimizer, train=False,
           epoch=epoch, label="Conjoined", eval=True)
train_test(test_sequences, model=disjoint_model,
           optimizer=disjoint_optimizer, train=False,
           epoch=epoch, label="Disjoined", eval=True)
