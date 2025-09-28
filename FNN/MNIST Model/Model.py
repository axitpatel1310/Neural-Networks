"""
import torch
from torchvision import datasets, transforms
train_ds = datasets.MNIST(root='/data', download=True, train=True, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds))
data,target = next(iter(loader))
mean = data.mean().item()
std = data.std().item()
print(mean,std)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

Batch_size = 128
Lr = 1e-3
epoches = 10

transform_ = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x.to(torch.float32)
])

train_ds = datasets.MNIST(root='/data',download=True,train=True,transform=transform_)
test_ds = datasets.MNIST(root='/data',download=False,train=False,transform=transform_)

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=Batch_size)
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=Batch_size)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

loss_ = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=Lr)

model.train()

for epoch in range(1,epoches+1):
    correct = 0 
    total = 0
    for x,y in train_loader:
        output = model(x)
        loss = loss_(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = output.argmax(dim=1)
        correct += (preds == y).sum().item()
        total = y.size(0)
    acc = correct * 100/total
    print(f"Epoch [{epoch}/{epoches}] Accuracy: {acc:.2f}")

model.eval()

test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for x,y in test_loader:
        output = model(x)
        loss = loss_(output,y)
        test_loss += loss
        
        preds = output.argmax(dim=1)
        correct += (preds == y).sum().item()
        total = y.size(0)
        
avg_loss = test_loss/len(test_loader)
accuracy = 100* correct / total

print(avg_loss, accuracy)    