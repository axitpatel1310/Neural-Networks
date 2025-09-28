import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer()
x = df.data
y = df.target

epoches = 50
BatchSize = 32
lr = 1e-3

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)

train_ds = torch.utils.data.TensorDataset(x_train,y_train)
test_ds = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BatchSize)
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=BatchSize)

model = nn.Sequential(
    nn.Linear(30,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,1),
    nn.Sigmoid()
)

loss_ = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

for epoch in range(1,epoches+1):
    
    model.train()
    
    total,running_loss,correct = 0,0,0
    for x,y in train_loader:
        output = model(x)
        loss = loss_(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss = loss.item() * x.size(0)
        correct = (output > 0.5).float()
        total = y.size(0)
    
    total_accuracy = correct / total
    total_loss = running_loss / total

    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            output = model(x)
            preds = (output > 0.5).float()
            test_acc = (preds == y).float().mean().item()
            
    print(f"Trainning Accuracy {total_accuracy}, Loss {total_loss}")
    print(f"Test Accuracy: {test_acc}")