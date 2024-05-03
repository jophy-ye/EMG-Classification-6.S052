import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
import re
from dataset import EMGDataset, data_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----------------------------
# Audio Classification Model
# ----------------------------
# should have 
# X = 129
# Y = 532 
# channels = 256
# as input layer
class AudioClassifier(nn.Module):
    # Build the model architecture
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(256, 128, kernel_size=(9, 9), stride=(2, 2), padding=(4, 4))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=2)
        self.activation = nn.Softmax()

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
        x = self.activation(self.lin(x))

        # Final output
        return x

# Create the model and put it on the GPU if available
model = AudioClassifier()
device = (
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
device = torch.device(device)
model = model.to(device)
# Check that it is on Cuda
next(model.parameters()).device

train_dataset = EMGDataset('emg_dataset', mode="train", transform=data_transform['train'])
val_dataset = EMGDataset('emg_dataset', mode="test", transform=data_transform['test'])

trainDataLoader = DataLoader(train_dataset, batch_size=1, shuffle=True)#, num_workers=5, pin_memory=True)
valDataLoader = DataLoader(val_dataset, batch_size=1, shuffle=False)#, num_workers=5, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Concatenate the input tensors along the channel dimension
num_epochs = 800    # !!! main adjustable parameter !!!
losses = []
for epoch in tqdm(range(num_epochs)):
    for ix in range(len(train_dataset)):
        input_tensor, label = next(iter(trainDataLoader))
        input_tensor  = input_tensor.to(device)
        label = label.to(device)
        # Forward pass
        output = model(input_tensor)

        # Compute the loss
        loss = criterion(output, label)
        losses.append(loss)

        # Backward pass: Compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

    if epoch % 20 == 0:
        print(f'On epoch {epoch}: loss={losses[-1]}')
    if epoch % 200 == 0:    # save periodically
        torch.save(model.state_dict(), 'parameters_task.pt')

torch.save(model.state_dict(), 'parameters_task.pt')