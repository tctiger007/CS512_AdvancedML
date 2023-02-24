import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = "../Figures/"
def plot_accuracy(epoch, test, file_name, train = None):
# Data
    x= [x for x in range(epoch)]
    w = 10
    h = 8
    d = 100
    plt.legend()
    plt.figure(figsize=(w, h), dpi=d)
    if train:
        plt.plot(x, train, color='red', linewidth=2, linestyle = 'dashed', label="Train")
    plt.plot(x, test, color = 'blue', linewidth = 2, label='Test')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
    legend.get_frame()
    plt.ylim((0, 110))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(path + file_name)

def plot_epsilon(epoch, epoch_epsilon, ep_opt, test_opt, test_ep1, test_ep2, test_ep3, ep1, ep2, ep3, file_name):
# Data
    x= [x for x in range(epoch)]
    x_ep = [x_ep for x_ep in range(epoch_epsilon)]
    w = 10
    h = 8
    d = 100
    path = "../Figures/"
    plt.legend()
    plt.figure(figsize=(w, h), dpi=d)
    plt.plot(x, test_opt, color='red', linewidth=2, label="Optimal Epsilon = " + str(ep_opt))
    plt.plot(x_ep, test_ep1, color = 'blue', linewidth = 2, linestyle = 'dashed', label=ep1)
    plt.plot(x_ep, test_ep2, color = 'green', linewidth = 2, linestyle = 'dashdot', label=ep2)
    plt.plot(x_ep, test_ep3, color = 'magenta', linewidth = 2, linestyle = 'dotted', label=ep3)
    legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
    legend.get_frame()
    plt.ylim((0, 110))
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.savefig(path + file_name)

basic_epoch = 100 
Adv_epoch = 100
Prox_epoch = 100
test_acc_basic = []
train_acc_basic = []
test_acc_adv = []
test_acc_prox = []

with open(path + "BasicModel_train.txt", "r") as f:
  for line in f:
    train_acc_basic.append(float(line.strip()))

with open(path + "BasicModel_test.txt", "r") as f:
  for line in f:
    test_acc_basic.append(float(line.strip()))

## plot test and train accuaracy. P2 
plot_accuracy(epoch = basic_epoch, test = test_acc_basic, file_name = 'BasicModel.png', train = train_acc_basic)


with open(path + "AdvModel_acc.txt", "r") as f:
  for line in f:
    test_acc_adv.append(float(line.strip()))

plot_epsilon(epoch = Adv_epoch, epoch_epsilon = 50, ep_opt = 0.08, test_opt = test_acc_adv[:100], 
    test_ep1 = test_acc_adv[100:100 + 50], test_ep2 = test_acc_adv[200:200 + 50], test_ep3 = test_acc_adv[300:300+50], 
    ep1 = 0.01, ep2 = 0.1, ep3 = 1.0, file_name = 'AdvModel.png')

## plot test accuracy of prox_model 
with open(path + "ProxModel_acc.txt", "r") as f:
  for line in f:
    test_acc_prox.append(float(line.strip()))

plot_epsilon(epoch = Prox_epoch, epoch_epsilon = 50, ep_opt = 0.08, test_opt = test_acc_prox[:100], 
    test_ep1 = test_acc_prox[100:100 + 50], test_ep2 = test_acc_prox[200:200 + 50], test_ep3 = test_acc_prox[300:300+50], 
    ep1 = 0.1, ep2 = 1.0, ep3 = 5.0, file_name = 'ProxModel.png')



