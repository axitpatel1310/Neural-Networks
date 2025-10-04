import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

"""
train_ds = datasets.CIFAR10(root='data',download=True,train=True,transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(train_ds,batch_size=256)
data = next(iter(loader))[0]
mean = data.mean(dim=[0,2,3])
std = data.std(dim=[0,2,3])
print(mean,std)

# tensor([0.4869, 0.4771, 0.4391]) 
# tensor([0.2443, 0.2402, 0.2525])
"""
mean = (0.4869, 0.4771, 0.4391)
std = (0.2443, 0.2402, 0.2525)
BatchSize = 128

trs = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean,std)
])

train_ds = datasets.CIFAR10(root='data',download=True,train=True,transform=trs)
test_ds = datasets.CIFAR10(root='data',download=True,train=False,transform=trs)

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BatchSize,shuffle=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=256,shuffle=False,num_workers=2)

model = nn.Sequential(
    nn.Conv2d(3,64,3,padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    
    nn.Conv2d(64,64,3,padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    
    nn.MaxPool2d(2),
    
    nn.Conv2d(64,128,3,padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    
    nn.Conv2d(128,128,3,padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    
    nn.MaxPool2d(2),
    
    nn.Flatten(),
    nn.Linear(128*8*8,256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256,10)
)

optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)

def acc(dl):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in dl:
            p = model(x).argmax(1)
            correct += (p==y).sum().item()
            total += y.size(0)
    return correct/total

best,best_state = 0,None
for epoch in range(10):
    model.train()
    for x,y in train_loader:
        loss = f.cross_entropy(model(x),y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sched.step()
    te = acc(test_loader)
    if te > best:
        best,best_state = te, {k:v.detach().clone() for k,v in model.state_dict().items()}
    print(f"epoch {epoch+1} | test_acc = {te:.4f}")

model.load_state_dict(best_state)
print(best)

