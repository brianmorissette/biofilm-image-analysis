import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
   def __init__(self, in_channels, num_classes, image_size):
       super().__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear(16 * (image_size // 4) * (image_size // 4), num_classes) # 7 * 7 for 28x28 images that were pooled twice down to 7x7

   def forward(self, x):
       x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x

