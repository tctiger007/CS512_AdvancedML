import time
import torch
import torch.optim as optim
import torch.utils.data as data_utils

from data_loader import get_dataset
import matplotlib.pyplot as plt
import numpy as np
from crf import CRF


# Tunable parameters
batch_size = 64
num_epochs = 10
max_iters  = 1000
# print_iter = 25 # Prints results every n iterations
conv_shapes = [[1,64,128]] #


# Model parameters
input_dim = 128
embed_dim = 128
num_labels = 26
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the CRF model
crf_model = CRF(input_dim, embed_dim, conv_shapes, num_labels, batch_size).to(device)

# Setup the optimizer
opt = optim.LBFGS(crf_model.parameters(), lr = 0.1)


##################################################
# Begin training
##################################################
step = 0

# Fetch dataset
dataset = get_dataset()
split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).float())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).float())

# print(len(train[0][1][0]))
letterwise_train = []
letterwise_test = []
wordwise_train = []
wordwise_test = []
         
for i in range(num_epochs):
    print("\n--------------Starting Epoch {}-------------------".format(i))
    start_epoch = time.time()
    # Define train and test loaders
    train_loader = data_utils.DataLoader(train,  # dataset to load from
                                            batch_size=batch_size,  # examples per batch (default: 1)
                                            shuffle=True,
                                            sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                            num_workers=5,  # subprocesses to use for sampling
                                            pin_memory=False,  # whether to return an item pinned to GPU
                                            )

    test_loader = data_utils.DataLoader(test,  # dataset to load from
                                        batch_size=batch_size,  # examples per batch (default: 1)
                                        shuffle=False,
                                        sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                        num_workers=5,  # subprocesses to use for sampling
                                        pin_memory=False,  # whether to return an item pinned to GPU
                                        )
    print('Loaded dataset... ')
    avg_word_acc_train = 0
    avg_word_acc_test = 0
    avg_letter_acc_train = 0
    avg_letter_acc_test = 0
    # Now start training
    for i_batch, sample in enumerate(train_loader):
        print("\n----- Starting Epoch-{} Batch-{} ------".format(i,i_batch))
        start_batch = time.time()
        train_X = sample[0].to(device)
        train_Y = sample[1].to(device)
        print(train_Y.dtype)
        
        # compute loss, grads, updates:
        def closure() :
            opt.zero_grad() # clear the gradients
            tr_loss = crf_model.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
            tr_loss.backward()
            # tr_loss.backward(train_Y) # Run backward pass and accumulate gradients
            return tr_loss

        start_opt_step = time.time()
        opt.step(closure) # Perform optimization step (weight updates)
        print("OPT step Epoch-{} Batch-{} Step-{} TIME ELAPSED = {}".format(i,i_batch,step,time.time() - start_opt_step))
        for name, param in crf_model.named_parameters():
            if param.requires_grad:
                print ("Params after OPT step ",name, param.data)
        
        ##################################################################
        # IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
        ##################################################################
        print("Starting Accuracy Calculation ....")

        random_ixs = np.random.choice(test_data.shape[0], batch_size, replace=False)
        test_X = test_data[random_ixs, :]
        test_Y = test_target[random_ixs, :]

        # Convert to torch
        test_X = torch.from_numpy(test_X).float().to(device)
        test_Y = torch.from_numpy(test_Y).long().to(device)
        
        total_train_letters = torch.sum(train_Y).item()
        total_test_letters = torch.sum(test_Y).item()
        total_train_words = len(train_Y)
        total_test_words = len(test_Y)

        with torch.no_grad() :
            print("Getting Training predictions...")
            train_predictions = crf_model(train_X)
            print("Getting Test predictions...")
            test_predictions = crf_model(test_X)
        train_word_acc = 0
        train_letter_acc = 0
        for y,y_predict in zip(train_Y,train_predictions) :
            num_letters = int(torch.sum(y).item())                      ## Number of letters in the word
            if (torch.all(torch.eq(y[:num_letters], y_predict[:num_letters]))) :      ## if all letters are predicted correct
                train_word_acc += 1
            train_letter_acc += num_letters - (((~torch.eq(y[:num_letters],y_predict[:num_letters])).sum()) / 2).item()
        test_word_acc = 0
        test_letter_acc = 0
        for y,y_predict in zip(test_Y,test_predictions) :
            num_letters = int(torch.sum(y).item())                      ## Number of letters in the word
            if (torch.all(torch.eq(y[:num_letters], y_predict[:num_letters]))):      ## if all letters are predicted correct
                test_word_acc += 1
            test_letter_acc += num_letters - (((~torch.eq(y[:num_letters],y_predict[:num_letters])).sum()) / 2).item()
        ## Calculate accuracies for current batch (after current update step)
        train_letter_acc /= total_train_letters
        test_letter_acc /= total_test_letters
        train_word_acc /= total_train_words
        test_word_acc /= total_test_words
        
        ## collect accuracies for 100 steps
        letterwise_train.append(train_letter_acc)
        letterwise_test.append(test_letter_acc)
        wordwise_train.append(train_word_acc)
        wordwise_test.append(test_word_acc)
        print("\nTraining Accuracy : ")
        print("\tword accuracy = ",wordwise_train)
        print("\tletter accuracy = ",letterwise_train)
        print("Test Accuracy : ")
        print("\tword accuracy = ",wordwise_test)
        print("\tletter accuracy = ",letterwise_test)

        ## Calculate average accuracies from first step till the current step
        avg_word_acc_train = sum(wordwise_train) / len(wordwise_train)
        avg_word_acc_test = sum(wordwise_test) / len(wordwise_test)
        avg_letter_acc_train = sum(letterwise_train) / len(letterwise_train)
        avg_letter_acc_test = sum(letterwise_test) / len(letterwise_test)
    
        print("\n avg_word_acc_train = {}\n avg_word_acc_test = {}\n avg_letter_acc_train = {}\n avg_letter_acc_test = {}\n".format(avg_word_acc_train,avg_word_acc_test,avg_letter_acc_train,avg_letter_acc_test))    
        print("Batch completed Epoch-{} Batch-{} Step-{} TIME ELAPSED = {}".format(i,i_batch,step,time.time() - start_batch))
        step += 1
        if step > max_iters: raise StopIteration
    
    print("Epoch completed Epoch-{} Batch-{} Step-{} TIME ELAPSED = {}".format(i,i_batch,step-1,time.time() - start_epoch))

# print("\nTraining Accuracy : ")
# print("\tword accuracy = ",wordwise_train)
# print("\tletter accuracy = ",letterwise_train)
# print("Test Accuracy : ")
# print("\tword accuracy = ",wordwise_test)
# print("\tletter accuracy = ",letterwise_test)



# x = np.arange(1,101)
# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.plot(x,letterwise_train, label = "Batch Training Accuracy")
# ax1.plot(x,letterwise_test, label = "Batch Test Accuracy")
# ax1.set_title("Letterwise Accuracy")
# ax1.set_xlabel("Batches")
# ax1.set_ylabel("Accuracy")
# ax2.plot(x,wordwise_train, label = "Batch Training Accuracy")
# ax2.plot(x,wordwise_test, label = "Batch Test Accuracy")
# ax2.set_title("Wordwise Accuracy")
# ax2.set_xlabel("Batches")
# ax2.set_ylabel("Accuracy")
# plt.show()
            
### TODO : plot letterwise accuracy for training and test using letterwise_train and letterwise_test

### TODO : plot wordwise accuracy for training and test using wordwise_train and wordwise_test
