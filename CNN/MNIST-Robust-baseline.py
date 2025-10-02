"""
from torchvision import transforms, datasets
dataset = datasets.MNIST(root='data',download=True,transform=transforms.ToTensor())
x = dataset.data.float() / 255.0
y = dataset.targets
mean = x.mean()
std = x.std()
print(mean.item())
print(std.item())
"""
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

tfr = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(10,translate=(0.05,0.05),scale=(0.95,1.05)),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_ds = datasets.MNIST(root='data',download=True,train=True,transform=tfr)
n_val = 6000 #(10% of the total dataset [mnist have 60,000 images])
train_ds,val_ds = torch.utils.data.random_split(full_ds,[len(full_ds)-n_val,n_val])
val_ds.dataset.trasform = tfr
test_ds = datasets.MNIST(root='data',download=True,train=False,transform=tfr)

train_dl = torch.utils.data.DataLoader(train_ds,batch_size=128,shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds,batch_size=256)
val_dl = torch.utils.data.DataLoader(val_ds,batch_size=256)

model = nn.Sequential(
    nn.Conv2d(1,8,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    nn.Conv2d(8,16,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    nn.Flatten(),
    nn.Linear(16*7*7,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

optimizer = optim.Adam(model.parameters(),lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.5)

def acc(dl):
    model.eval();
    correct = total = 0
    with torch.no_grad():
        for x,y in dl:
            p = model(x).argmax(1)
            correct += (p==y).sum().item()
            total = y.size(0)
    return correct/total

best, best_state = 0.0,None
for epoch in range(5):
    model.train()
    for x,y in train_dl:
        loss = f.cross_entropy(model(x),y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    va = acc(val_dl)
    sched.step()
    if va > best:
        best,best_state = va, {k:v.detach().clone() for k,v in model.state_dict().items()}
    print(f"Epoch: {epoch+1}/5 | val_acc= {va:.4f}")

model.load_state_dict(best_state)
print(acc(test_dl))