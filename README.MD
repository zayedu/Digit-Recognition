# Handwritten Digit Classification

## Overview

Hey there! Welcome to my handwritten digit classification project. In this project, I've developed a machine learning model using the MNIST dataset and PyTorch to classify handwritten digits. It's a fascinating journey from loading the data, preprocessing it, building a convolutional neural network (CNN), to training the model and making predictions. Let's dive in!

## Dataset

The project uses the famous MNIST dataset, which is a large collection of handwritten digits. It's a go-to dataset for anyone delving into image processing systems. Here's how I load and transform this dataset into a tensor format, making it ready for use with PyTorch:

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    transform=ToTensor(),
    download=True
)
```

## Data Loading

Efficiency is key! I set up DataLoaders for both the training and test sets. This helps in efficiently loading the data during both the training and testing phases.

```python
from torch.utils.data import DataLoader

loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
}
```

## Model Architecture

Let's talk about the model. It's a CNN with two convolutional layers (to capture those patterns in the images), dropout (to avoid overfitting), and two fully connected layers. The final touch is the softmax function in the output layer, perfect for multi-class classification.

```python
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)
```

## Training and Evaluation

Training the model is where the magic happens. I use the Adam optimizer and cross-entropy loss function. Here's a peek into how the model is trained and evaluated:

```python
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train and test functions are defined here...
```

## Model Prediction

Finally, it's time to see the model in action! Here's how I make a sample prediction using the trained model, showcasing its ability to classify individual handwritten digits.

```python
import matplotlib.pyplot as plt

model.eval()
data, target = test_data[10]
data = data.unsqueeze(0).to(device)
output = model(data)
pred = output.argmax(dim=1, keepdim=True).item()
print(f'Prediction: {pred}')
image = data.squeeze(0).squeeze(0).cpu().numpy()
plt.imshow(image, cmap='gray')
plt.show()
```

## Contact

- Email: [umerz@mcmaster.ca](mailto:umerz@mcmaster.ca)
- LinkedIn: [linkedin.com/in/zayed-umer](https://linkedin.com/in/zayed-umer)
- GitHub: [github.com/zayedu](https://github.com/zayedu)
- Portfolio: [notzayed.me](http://notzayed.me/)
