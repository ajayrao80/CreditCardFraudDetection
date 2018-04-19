import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data = pd.read_csv("creditcard.csv")
data = np.array(data)

X_data = data[:, 0:-1]
y_data = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)


class CreditCardTrainDataLoader(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(X_train)
        self.x_data.type(torch.FloatTensor)
        self.y_data = torch.from_numpy(y_train)
        self.y_data.type(torch.FloatTensor)
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class CreditCardTestDataLoader(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(X_test)
        self.x_data.type(torch.FloatTensor)
        self.y_data = torch.from_numpy(y_test)
        self.y_data.type(torch.FloatTensor)
        self.len = X_test.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

batch_size = 50

data_set = CreditCardTrainDataLoader()
train_loader = DataLoader(
    dataset=data_set,
    batch_size=batch_size,
    shuffle=True
)

test_data_set = CreditCardTestDataLoader()
test_loader = DataLoader(
    dataset=test_data_set,
    batch_size=1,
    shuffle=False
)


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.sigmoid(out)
        out = self.layer2(out)
        return out


input_dim = 30
hidden_dim = 32
num_of_classes = 2

model = Model(input_dim, hidden_dim, num_of_classes)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
epochs = 5

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = Variable(inputs.type(torch.FloatTensor))
        labels = Variable(labels.type(torch.FloatTensor))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels.type(torch.LongTensor))
        loss.backward()
        optimizer.step()

        if(i+1) % 100 == 0:
            total = 0
            correct = 0
            for ins, l in test_loader:
                ins = Variable(ins.type(torch.FloatTensor))
                outs = model(ins)
                _, predicted = torch.max(outs.data, 1)

                total += l.size(0)
                equal = predicted.type(torch.IntTensor) == l.type(torch.IntTensor)
                if int(equal) == 1:
                    correct += 1

            accuracy = 100 * correct/total
            print("iteration ", i+1, " in Epoch ", epoch, " Loss : ", loss.data[0], " Accuracy: ", accuracy)


    


