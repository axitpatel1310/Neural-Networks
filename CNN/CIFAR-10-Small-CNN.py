import torch
from torchvision import datasets, transforms
import torch.nn.functional as f
import torch.nn as nn
import torch.optim as optim

"""
dataset = datasets.CIFAR10(root='cifer',download=True,transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset)
data = next(iter(loader))[0]
mean = data.mean(dim=[0,2,3])
std = data.std(dim=[0,2,3])
print(mean)
print(std)

tensor([0.5537, 0.4122, 0.2511])
tensor([0.1595, 0.1665, 0.1603])
"""

mean = (0.5537, 0.4122, 0.2511)
std = (0.1595, 0.1665, 0.1603)

trs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
    transforms.RandomHorizontalFlip()
])

train_ds = datasets.CIFAR10(root='cifer',download=True,train=True,transform=trs)
test_ds = datasets.CIFAR10(root='cifer',download=True,train=False,transform=trs)

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=256,)

model = nn.Sequential(
    nn.Conv2d(3,32,3,padding=1),
    nn.ReLU(),
    nn.Conv2d(32,32,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    nn.Conv2d(32,64,3,padding=1),
    nn.ReLU(),
    nn.Conv2d(64,64,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64*8*8,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

optimizer = optim.Adam(model.parameters(),lr=1e-3)

def acc(dl):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in dl:
            p = model(x).argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
    return correct/total

model.train()
epoches = 5

for epoch in range(1,epoches+1):
    for x,y in train_loader:
        loss = f.cross_entropy(model(x),y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epocches: {epoch}/{epoches} | Test Accuracy: {acc(test_loader):.4f}")

#76% accuracy