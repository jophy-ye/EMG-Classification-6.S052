import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
import re
from dataset import EMGDataset, data_transform

# ----------------------------
# Audio Classification Model
# ----------------------------
# should have 
# X = 129
# Y = 532 
# as input layer
class AudioClassifier (nn.Module):
    # Build the model architecture
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = (
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
device = torch.device(device)
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device
# load a pretrained ResNet model

train_dataset = EMGDataset('emg_dataset', mode="train", transform=data_transform['train'])
val_dataset = EMGDataset('emg_dataset', mode="test", transform=data_transform['test'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(myModel.parameters(), lr=0.01)

input_tensors = []  # List of tensors, each one has dim (num_channels, height, width), length of list is 124 patients
labels = []  # List of true labels

# Concatenate the input tensors along the channel dimension
num_epochs = 2000
for epoch in range(num_epochs):
    for ix in range(len(input_tensors)):
        input_tensor = input_tensors[ix]
        # Forward pass
        output = myModel(input_tensor)

        # Compute the loss
        loss = criterion(output, labels[ix])

        # Backward pass: Compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

