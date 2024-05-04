import re
from dataset import EMGDataset, data_transform
from torch.utils.data import DataLoader
import torch.optim as optim
from cnn import *
import os

if __name__ == '__main__':
    #train_dataset = EMGDataset('emg_dataset', mode="train", transform=data_transform['train'])
    #val_dataset = EMGDataset('emg_dataset', mode="test", transform=data_transform['test'])
    train_dataset = EMGDataset('np_emg_dataset', format="np", mode="train")
    val_dataset = EMGDataset('np_emg_dataset', format="np", mode="test")
    batchsize = 32
    trainDataLoader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,pin_memory=True) # num_workers=20, 
    valDataLoader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False,  pin_memory=True) #num_workers=20,

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

    num_epochs = 10000    # !!! main adjustable parameter !!!
    start_epoch = 0
    epoch_stamp = 500 # create model files every this many epoches
    modelweights_directory = "modelweights"
    losses_directory = "losses"
    os.makedirs(modelweights_directory, exist_ok=True)
    os.makedirs(losses_directory, exist_ok=True)

    # this is a model file to load
    loadfile = os.join(modelweights_directory, f"03-05-24-10_54__epoch_{999}__parameters_task_full.pt") #"03-05-24-08_00__parameters_task_full.pt"#"parameters_task.pt"

    if loadfile:
        checkpoint = torch.load(loadfile)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
   
    # the 
    #train(trainDataLoader = trainDataLoader, model = model, criterion = criterion, optimizer = optimizer, num_epochs=num_epochs, start_epoch=start_epoch, epoch_stamp=epoch_stamp, device=device, losses_directory= losses_directory, modelweights_directory=modelweights_directory)
    
    # #the following is code to load model files and potentially do something with them
    # weightfiles = [f"03-05-24-10_54__epoch_{epoch}__parameters_task_full.pt" for epoch in range(100, 1000, 100)]
    # weightfiles.append(f"03-05-24-10_54__epoch_{999}__parameters_task_full.pt")
    # print(weightfiles)
    # for weightfile in weightfiles:
    #     checkpoint = torch.load(os.join(modelweights_directory, weightfile))
    #     #model.load_state_dict(checkpoint)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']
    #     validate(valDataLoader = valDataLoader, model=model, criterion = criterion, device=device)

    validate(valDataLoader = valDataLoader, model=model, criterion = criterion, device=device)
    validate(valDataLoader = trainDataLoader, model=model, criterion = criterion, device=device)