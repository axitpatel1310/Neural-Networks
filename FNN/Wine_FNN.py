import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

BatchSize = 32
epoches = 50

df = load_wine()
x = df.data
y = df.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.long)
y_test = torch.tensor(y_test,dtype=torch.long)

train_ds = torch.utils.data.TensorDataset(x_train,y_train)
test_ds = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BatchSize)
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=BatchSize)

model = nn.Sequential(
    nn.Linear(13,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,3)
)

loss_ = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(1,epoches+1):
    total,correct,running_loss = 0,0,0
    for x, y in train_loader:
        output = model(x)
        loss = loss_(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        preds = output.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    train_acc = correct / total
    train_loss = running_loss / total    
    
    model.eval()
    
    with torch.no_grad():
        for x,y in test_loader:
            output = model(x)
            preds = output.argmax(dim=1)
            test_acc = (preds == y).float().mean().item()
            
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{epoches} | Trainning Accuracy: {train_acc:.4f} | Trainning Loss: {train_loss:.4f}]")
        print(f"Test Accuracy: {test_acc}")