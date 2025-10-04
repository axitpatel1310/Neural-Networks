import torch
from torchvision import datasets,transforms,models
import torch.nn as nn
"""
dataset = datasets.CIFAR10(root='data',download=True,train=True,transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset,batch_size=256)
data = next(iter(loader))[0]
mean = data.mean(dim=(0,2,3))
std = data.std(dim=(0,2,3))
print(mean)
print(std)
#tensor([0.4869, 0.4771, 0.4391])
#tensor([0.2443, 0.2402, 0.2525])
"""
mean = (0.4869, 0.4771, 0.4391)
std = (0.2443, 0.2402, 0.2525)

def main():
    model_path = r'cifar10.pth'
    trs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    test_ds = datasets.CIFAR10(root='data',download=True,train=False,transform=trs)
    test_loader = torch.utils.data.DataLoader(test_ds,batch_size=256)
    model = models.resnet18(num_classes=10)
    state = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(state)
    print("Loaded Checkpoints")
    model.eval()
    correct,total = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        acc = (correct/total)*100
        print(f"Accuracy : {acc:.4f} %")
        
if __name__ == '__main__':
    main()