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
from datetime import datetime


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


#train_dataset = EMGDataset('emg_dataset', mode="train", transform=data_transform['train'])
#val_dataset = EMGDataset('emg_dataset', mode="test", transform=data_transform['test'])

train_dataset = EMGDataset('np_emg_dataset', format="np", mode="train")
val_dataset = EMGDataset('np_emg_dataset', format="np", mode="test")
batchsize = 32
trainDataLoader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)#, num_workers=20, pin_memory=True)
valDataLoader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)#, num_workers=20, pin_memory=True)

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Concatenate the input tensors along the channel dimension


import numpy as np
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def train(num_epochs):
    # set the model to train mode
    model.train()
    losses = []
    running_loss = 0.0
    total_loss = 0.0
    running_count = 0
    total_count = 0

    filename = datetime.now().strftime('%d-%m-%y-%H_%M__parameters_task.pt')   
    for epoch in tqdm(range(num_epochs)):
        for batch_index, (inputs, labels) in enumerate(trainDataLoader):
            inputs  = inputs.to(device)
            labels = labels.to(device)
            

            # Forward pass
            output = model(inputs)

            # Compute the loss
            loss = criterion(output, labels)
            

            # Backward pass: Compute gradients
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # update loss and count
            running_loss += loss.item() * inputs.size(0)
            total_loss += loss.item() * inputs.size(0)

            running_count += inputs.size(0)
            total_count += inputs.size(0)
            # print every 50 mini-batches
            if batch_index % 50 == 49:
                print('[%d, %5d] avg batch loss: %.3f avg epoch loss: %.3f' %
                    (epoch + 1, batch_index + 1, running_loss / running_count, total_loss / total_count))
                running_loss = 0.0
                running_count = 0
        losses.append(loss.item())
        if epoch % 20 == 0:
            print(f'On epoch {epoch}: loss={losses[-1]}')
        if epoch % 200 == 0:    # save periodically
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, filename)
    print(losses)
    print(moving_average(losses, 5))
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, filename)

def validate(loadfile=None):
    # set the model to evaluation mode
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # no need to track gradients for validation
    with torch.no_grad():
        for inputs, labels in valDataLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # TODO: from your output (which are probabilities for each class, find the predicted
            # class)
            print(outputs)
            classifications = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            correct_count = 0
            print(labels)
            for idx, classification in enumerate(classifications):
                correct_count += classification == labels[idx]
            #correct_count = classifications[classifications == labels]

            # update loss and count
            total_loss += loss.item() * labels.size(0)
            total_correct += correct_count
            total_count += labels.size(0)

    accuracy = 100 * total_correct / total_count
    print(accuracy)
    print(total_correct)
    print(total_count)
    print()
    print(f"Evaluation loss: {total_loss / total_count :.3f}")
    print(f'Accuracy of the model on the validation images: {accuracy: .2f}%')
    print()


if __name__ == '__main__':
    num_epochs = 20000    # !!! main adjustable parameter !!!
    loadfile = None#"parameters_task.pt"
    if loadfile:
        checkpoint = torch.load(loadfile)
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
   
    train(num_epochs=num_epochs)
    validate()