# EMG-Classification-6.S052
Class final project for MIT 6.S052 Spring 2024 on classifying EMG signals

# sampling rate
2048 from Muedit (calculated 2000 from  S1_10_DF.otb%2B_decomp.mat_edited.mat)

# number of electrodes
4 grids of 64 electrode for KE (256)
265 for DF

# some information about the files
- cnn.py is more like a library
- run_model to run the model, this is most likely going to be custom to
    your run so you can keep a local copy that is not pushed up to git
- exploration.ipynb, like run_model you can keep a local version of
    this, this what I used to understand the dataset

# to run model
look at run model file, 
Weights in model weights, pick the one you would like, 
the best that I have found is already selected

test_files.txt -> these are the only file you would need to run test. You can take them from  np_emp_dataset

make sure you are only loading these files, by commenting out the dataset and dataloader for training

# losses
these corrospond to losses while training the model. Each file contains a list. there is one entry for every epoch in the list. The timestamp in the names of the files matches the training run and corrosponding set of modelweights.


## key thing to remember
model was trained until 1000 epoch, 
then a second run starting at 1000 epoch upto 10k was run.
the losses that corrospond to this training history are split into two files.
03-05-24-10_54__losses.txt
03-05-24-13_23__losses.txt

## dropbox link
https://www.dropbox.com/scl/fo/euwnqathf6uu8pgzvodw1/ACnbilEvk7C3sS9n0f5ujbs?rlkey=b3nrpf0a3sytgfst6phitaf7l&dl=0
