from torchvision import transforms
from PIL import Image
import torch
from Model import model

img = Image.open('image.jpg').convert('L')

transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    lambda x: x.to(torch.float32)
])

img = transform(img).unsqueeze(0)

model.eval()

with torch.no_grad():
    output = model(img)
    preds = output.argmax(dim=1)

print(preds)