import torch
import torch.nn as nn
from torchvision import datasets,transforms
import torch.optim as optim
import torch.nn.functional as f

BatchSize = 32

train_ds = datasets.MNIST(root='data',download=True,train=True,transform=transforms.ToTensor())
test_ds = datasets.MNIST(root='data',download=True,train=False,transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BatchSize,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=BatchSize)

model = nn.Sequential(
    nn.Conv2d(1,8,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    nn.Flatten(),
    nn.Linear(16*7*7,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

optimizer = optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(5):
    model.train()
    for x,y in train_loader:
        output = model(x)
        loss = f.cross_entropy(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total = y.size(0)
    acc = correct/total
    print(f"epoch {epoch+1} | Test Accuracy {acc:.4f}")
    
#60% accuracy