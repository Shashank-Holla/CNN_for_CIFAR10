import torch
import torch.nn as nn
from train import train
from test import test
from model import Net
from metrics import train_test_metrics_graph
from transform import transform

import torch.optim as optim

EPOCHS =10
train_accuracy = []
test_accuracy = []
train_loss = []
test_loss = []

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

trainloader, testloader, classes = transform()

for epoch in range(EPOCHS):
  print("epoch:", epoch)
  train_accuracy_delta, train_loss_delta = train(net, device, trainloader, optimizer, criterion, epoch)
  test_accuracy_delta, test_loss_delta = test(net, device, testloader, criterion)
  train_accuracy.append(train_accuracy_delta)
  train_loss.append(train_loss_delta)
  test_accuracy.append(test_accuracy_delta)
  test_loss.append(test_loss_delta)
  
print(train_accuracy)
print(test_accuracy)

print(train_loss)
print(test_loss)
train_test_metrics_graph(train_accuracy, train_loss, test_accuracy, test_loss)