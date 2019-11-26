import torchvision
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from models import vgg
from torchvision import datasets, models, transforms
from catalyst.dl import utils
from catalyst.dl.runner import SupervisedRunner
from utils import progress_bar
import collections
import os

#constant
batch_size = 1
device = "cuda"
num_classes = 10
channels = 3
start_epoch = 1

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#MNIST_Dataset
ds_train = torchvision.datasets.CIFAR10(root='data',
                                        train=True,
                                        transform=transform_train,
                                        download=True)

ds_test = torchvision.datasets.CIFAR10(root='data',
                                        train=False,
                                        transform=transform_test)
                                        
dataloader_train = data.DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
dataloader_test = data.DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = vgg.VGG16(num_classes,channels).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader_train):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        print(inputs.shape)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader_train), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader_test):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader_test), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+10):
    train(epoch)
    test(epoch)

# catalyst_version
# # model runner
# runner = SupervisedRunner()

# # model training
# runner.train(
#     model=net,
#     criterion=criterion,
#     optimizer=optimizer,
#     loaders=loaders,
#     logdir=logdir,
#     num_epochs=num_epochs,
#     verbose=True
# )
