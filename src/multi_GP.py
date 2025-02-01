import numpy as np
import torch

from torch import nn, optim
import torch.nn.functional as F

device = torch.device("mps")
print("device:", device)

# Model structure for MNIST dataset
class ConvFeatNet(nn.Module):
    def __init__(self, out_size = 32):
        super(ConvFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, out_size)
        self.fc3 = nn.Linear(out_size, 10)
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        f = self.fc2(x)
        x = self.fc3(F.normalize(f, p=2, dim=1))
        return f, x # F.log_softmax(x, dim = 1)

class ConvFeatNet2(nn.Module):
    def __init__(self, out_size = 32):
        super(ConvFeatNet2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.BatchNorm2d(10),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.BatchNorm2d(20),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )

        
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, out_size)
        self.fc3 = nn.Linear(out_size, 10) 
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        f = self.fc2(x)
        x = self.fc3(f)
        return f, x


# parameter refers to k 
def train(network, trloader, epochs, learning_rate = 0.001, momentum = 0.5, verbal = False):
    # optimizer = 
    # optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum) , weight_decay=1e-4
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)  
    error = nn.CrossEntropyLoss() #label_smoothing=0.01

    network.to(device)
    network.train()

    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(trloader):
            data = data.to(device)
            target = target.to(device)
            data.requires_grad_(True)
        
            optimizer.zero_grad()
            _, output = network(data)

            # loss = F.nll_loss(output, target)
            loss = error(output, target)
            loss.backward()

            optimizer.step()

            if verbal and batch_idx % 400 == 0:
                print("epoch: ", epoch, ", batch: ", batch_idx, ", loss:", loss.item())

def scores(network, dataloader, is_test=False):
    network.eval()
    network.to(device)
    outputs, out_features, losses = [], [], []
    loss, correct = 0, 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            out_feature, output = network(data)
            outputs.append(output)
            out_features.append(out_feature)
            loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        loss /= len(dataloader.dataset)
        losses.append(loss)
        print('\nTest' if is_test else '\nTrain', ' set: Avg. loss :{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)))
        acc = correct / len(dataloader.dataset)
        out_features = torch.cat(out_features,0)
        outputs = torch.cat(outputs, 0)
        return out_features, outputs, acc.item()
