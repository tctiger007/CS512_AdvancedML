
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
import numpy as np
from Classifier import LSTMClassifier


# Hyperparameters, feel free to tune


batch_size = 27
output_size = 9   # number of class
hidden_size = 10  # LSTM output size of each time step
input_size = 12
basic_epoch = 100
Adv_epoch = 100
Prox_epoch = 100

if torch.cuda.is_available() :
    device = torch.device('cuda')
else :
    device = torch.device('cpu')


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)



# Training model
def train_model(model, train_iter, mode, prox_epsilon=1, epsilon = 0.01):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        input = batch[0]
        input.requires_grad = True
        target = batch[1]
        target = torch.autograd.Variable(target).long()
        r = 0
        optim.zero_grad()
        prediction = model(input, r, batch_size=input.size()[0], mode=mode, prox_epsilon=prox_epsilon)
        # print("prediction ", prediction.shape)
        # print("target ", target.shape)
        loss = loss_fn(prediction, target)
        if mode == 'AdvLSTM':
            ''' Add adversarial training term to loss'''
            r = compute_perturbation(loss, model)
            adv_prediction = model(input, r, batch_size=input.size()[0], mode=mode, prox_epsilon=prox_epsilon, epsilon=epsilon)
            loss = loss_fn(adv_prediction, target)


        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/(input.size()[0])
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


# Test model
def eval_model(model, test_iter, mode):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    r = 0
    # with torch.no_grad():
    for idx, batch in enumerate(test_iter):
        input = batch[0]
        target = batch[1]
        target = torch.autograd.Variable(target).long()
        prediction = model(input, r, batch_size=input.size()[0], mode = mode)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
        acc = 100.0 * num_corrects.double()/(input.size()[0])
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(test_iter), total_epoch_acc / len(test_iter)




def compute_perturbation(loss, model):
    '''need to be implemented'''
    # Use autograd
    g = grad(outputs=loss, inputs=model.get_lstm_input(), retain_graph=True, only_inputs=True, allow_unused=True)
    g = g[0]
    r = g / F.normalize(g)
    # print(g.shape, model.get_lstm_input().shape)
    # print(r)
    # input()
    return r
    # return the value of g / ||g||



''' Training basic model '''

train_iter, test_iter = load_data.load_data('./JV_data.mat', batch_size)


model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
loss_fn = F.cross_entropy
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)

train_acc_basic = []
test_acc_basic = []
for epoch in range(basic_epoch):
    train_loss, train_acc = train_model(model, train_iter, mode = 'plain')
    val_loss, val_acc = eval_model(model, test_iter, mode ='plain')
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
    train_acc_basic.append(train_acc)
    test_acc_basic.append(val_acc)

# # with open("../Figures/BasicModel_train.txt", "w") as f:
# #     for item in train_acc_basic:
# #         f.write("%s\n" % item) 

# # with open("../Figures/BasicModel_test.txt", "w") as f:
# #     for item in test_acc_basic:
# #         f.write("%s\n" % item) 

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


''' Part 3 '''
''' Save and Load model'''
model_PATH = "./myLSTMmodel.pth"

#1. Save the trained model from the basic LSTM
torch.save(model.state_dict(), model_PATH)

# 2. load the saved model to Prox_model, which is an instance of LSTMClassifier
Prox_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Prox_model.load_state_dict(torch.load(model_PATH, map_location = device))

# 3. load the saved model to Adv_model, which is an instance of LSTMClassifier
Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Adv_model.load_state_dict(torch.load(model_PATH, map_location = device))




# # ''' Training Adv_model'''
test_acc_adv = []
for epsilon in [0.08, 0.01, 0.1, 1.0]: # (optimal epsilon = 0.08) ep1 = 0.01, ep2 = 0.1 , ep3 = 1
##for epsilon in range(0, 10, 2):
##for epsilon in np.logspace(-5, 5, 11):
    Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
    for epoch in range(Adv_epoch):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=1e-3, weight_decay=1e-3)
        train_loss, train_acc = train_model(Adv_model, train_iter, mode = 'AdvLSTM', epsilon = epsilon)
        val_loss, val_acc = eval_model(Adv_model, test_iter, mode ='AdvLSTM')
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
        test_acc_adv.append(val_acc)
    print(f'Epsilon: {epsilon}, Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')

# # with open("../Figures/AdvModel_acc.txt", "w") as txt_file:
# #     for item in test_acc_adv:
# #         txt_file.write("%s\n" % item) 


''' Training Prox_model'''
test_acc_prox = []
for prox_epsilon in [0.08, 0.1, 1.0, 5.0]:
###for prox_epsilon in np.logspace(-2,3,6):
    Prox_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
    for epoch in range(Prox_epoch):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=1e-3, weight_decay=1e-3)
        train_loss, train_acc = train_model(Prox_model, train_iter, mode = 'ProxLSTM', prox_epsilon=prox_epsilon)
        val_loss, val_acc = eval_model(Prox_model, test_iter, mode ='ProxLSTM')
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
        test_acc_prox.append(val_acc)
    #print(f'Prox_epsilon: {prox_epsilon}, Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')

# # with open("../Figures/ProxModel_acc.txt", "w") as txt_file:
# #     for item in test_acc_prox:
# #         txt_file.write("%s\n" % item) 


#### Q6
## dropout layer
## To run: set self.apply_dropout = True in Classifier.py
test_acc_prox_dropout = []
Prox_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
for epoch in range(Prox_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=1e-3, weight_decay=1e-3)
    train_loss, train_acc = train_model(Prox_model, train_iter, mode = 'ProxLSTM', prox_epsilon=1)
    val_loss, val_acc = eval_model(Prox_model, test_iter, mode ='ProxLSTM')
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
    test_acc_prox_dropout.append(val_acc)


# # with open("../Figures/ProxModel_acc_dropout.txt", "w") as txt_file:
# #     for item in test_acc_prox_dropout:
# #         txt_file.write("%s\n" % item) 

## batch normalize 
## To run: set self.apply_batch_norm = True in Classifier.py
test_acc_prox_batchnorm = []
Prox_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
for epoch in range(Prox_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=1e-3, weight_decay=1e-3)
    train_loss, train_acc = train_model(Prox_model, train_iter, mode = 'ProxLSTM', prox_epsilon=1)
    val_loss, val_acc = eval_model(Prox_model, test_iter, mode ='ProxLSTM')
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
    test_acc_prox_batchnorm.append(val_acc)

# # with open("../Figures/ProxModel_acc_batchnorm.txt", "w") as txt_file:
# #     for item in test_acc_prox_batchnorm:
# #         txt_file.write("%s\n" % item) 


