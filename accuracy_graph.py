import re
from dataset import EMGDataset, data_transform
from torch.utils.data import DataLoader
import torch.optim as optim
from cnn import *
import os

if __name__ == '__main__':
    #train_dataset = EMGDataset('emg_dataset', mode="train", transform=data_transform['train'])
    #val_dataset = EMGDataset('emg_dataset', mode="test", transform=data_transform['test'])
    # train_dataset = EMGDataset('np_emg_dataset', format="np", mode="train")
    val_dataset = EMGDataset('np_emg_dataset', format="np", mode="test")
    batchsize = 32
    # trainDataLoader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,pin_memory=True) # num_workers=20,
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
    # # Check that it is on Cuda
    # next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #train(trainDataLoader = trainDataLoader, model = model, criterion = criterion, optimizer = optimizer, num_epochs=num_epochs, start_epoch=start_epoch, epoch_stamp=epoch_stamp, device=device, losses_directory= losses_directory, modelweights_directory=modelweights_directory)

    # #the following is code to load model files and potentially do something with them
    weightfiles = []
    # print("weightfile", weightfiles)
    # weightfiles_1 = [f"03-05-24-10_54__epoch_{epoch}__parameters_task_full.pt" for epoch in range(100, 1000, 100)]
    # weightfiles.append(f"03-05-24-10_54__epoch_{999}__parameters_task_full.pt")
    # weightfiles_2 = [f"03-05-24-13_23__epoch_{epoch}__parameters_task_full.pt" for epoch in range(1000, 10000, 500)]
    # weightfiles.extend(weightfiles_1)
    # weightfiles.extend(weightfiles_2)
    
    # weightfiles.append(f"03-05-24-13_23__epoch_{9999}__parameters_task_full.pt")
    weightfiles.append(f"03-05-24-13_23__epoch_{7500}__parameters_task_full.pt")

    print(weightfiles)
    modelweights_directory = "./modelweights_directory"
    for weightfile in weightfiles:
        if modelweights_directory is None:
            print("error, directory none")
        checkpoint = torch.load(os.path.join(modelweights_directory, weightfile), map_location=torch.device('cpu'))
        #model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        validate(valDataLoader = valDataLoader, model=model, criterion = criterion, device=device)

    validate(valDataLoader = valDataLoader, model=model, criterion = criterion, device=device)
    # validate(valDataLoader = trainDataLoader, model=model, criterion = criterion, device=device)

# loss = [0.680, 0.694, 0.692, 0.708, 0.704, 0.700, 0.712, 0.688, 0.690, 0.681, 0.680, 0.673, 0.687, 0.703, 0.697, 0.718, 0.721, 0.732, 0.701, 0.704, 0.714, 0.663, 0.676, 0.652, 0.669, 0.674, 0.689, 0.685, 0.676, 0.676]
# accuracy = [0.4545, 0.4848, 0.5455, 0.5455, 0.5152, 0.5455, 0.5455, 0.6061, 0.5758, 0.6364, 0.6364, 0.6667, 0.5758, 0.5758, 0.5758, 0.5455, 0.5455, 0.5455, 0.5455, 0.5455, 0.5758, 0.5758, 0.6061, 0.6970, 0.5758, 0.5758, 0.5758, 0.5758, 0.5758, 0.5758]
# epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 999, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 9999]