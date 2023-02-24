# Imports
import os
import numpy as np
import scipy.optimize as opt
import string
from numpy.linalg import norm
from decoder_WW import decoder

train_data_PATH = "../data/train.txt"
file = open(train_data_PATH,"r")
train_data = (file.read()).strip()
train_data = train_data.split('\n')         ## Training data as a list of individual letters' data

test_data_PATH = "../data/test.txt"
file = open(test_data_PATH,"r")
test_data = (file.read()).strip()
test_data = test_data.split('\n')           ## Test data as a list of individual letters' data

model_PATH = "../data/model.txt"
model = np.genfromtxt(model_PATH, delimiter = ' ')

def extract_words(word_list) :      ## Extract all words from the list
    ##### Extract Y and X from data ######
    ##### shape of X should be each letter in a row of size 128
    Y = []              ## array of all labels of all words
    X = []              ## array of pixel values of all words
    Yt = []             ## array of labels of letters for a single word
    Xt = []             ## array of pixel values of letters for a single word
    for i in range(len(word_list)) :
        letter = (word_list[i]).split()         ## extract data corresponding to a single letter
        Yt.append(ord(letter[1].lower())-97)    ## extract the label and convert it to integer
        Xt.append(np.array(list(map(int,letter[5:]))))      ## extract pixel values, convert them to integer 
                                                            ## and form a numpy array for data manipulation
        if (int(letter[2]) == -1) :             ## Check for end of word
            Y.append(Yt)
            X.append(np.array(Xt))
            Yt = []
            Xt = []
    return Y,X
def extract_parameters(model) :     ## Extract W and T from the model
    W = model[:128*26]
    T = model[128*26:]
    W = np.reshape(W, (26,128))                 ## W contains each Wy as a row of size 128
    T = np.reshape(T, (26,26))
    T = T.transpose()
    return W,T

def get_bwd_msg(Xt,W,T) :
    m = len(Xt)                         ## number of letters in the word
    b = np.zeros((m,26))                ## message backwards
    
    def f(s,y) :                        ## < Wy, Xts >
        return np.dot(W[y,:],Xt[s,:])   
    def g(i,j) :                        ## T[i,j]
        return T[i,j]                   

    for s in range(m-2,-1,-1) :         ## Calculate backward messages b
        for j in range(26) :            ## s goes from m-2 to 0. b[m-1,i] = 1 for all i from 0 to 25
            res = []
            for i in range(26) :
                res.append(f(s+1,i) + g(j,i) + b[s+1,i])
            max_value = max(res)
            res = max_value + np.log(sum(np.exp(np.array(res) - max_value)))
            b[s,j] = res
    return b

def get_fwd_msg(Xt,W,T) :
    m = len(Xt)                         ## number of letters in the word
    a = np.zeros((m,26))                ## message forward
    
    def f(s,y) :                        ## < Wy, Xts >
        return np.dot(W[y,:],Xt[s,:])   
    def g(i,j) :                        ## T[i,j]
        return T[i,j]                   

    for s in range(1,m,1) :             ## Calculate forward messages a
        for j in range(26) :
            res = []
            for i in range(26) :
                res.append(f(s-1,i) + g(i,j) + a[s-1,i])
            max_value = max(res)
            res = max_value + np.log(sum(np.exp(np.array(res) - max_value)))
            a[s,j] = res
    return a

def get_log_Zx(Xt,W,b) :
    log_Zx = 0                                      ## Zx sum over all yi for given training example
    res = []
    for i in range(26) :
        res.append(np.dot(W[i,:],Xt[0,:]) + b[0,i])   
    max_value = max(res)
    log_Zx = max_value + np.log(np.sum(np.exp(np.array(res) - max_value))) 
    return log_Zx

def get_log_posterior_Yt(Yt,Xt,W,T) :        ## Calculate log(p(Yt|Xt))
    def f(s,y) :                        ## < Wy, Xts >
        return np.dot(W[y,:],Xt[s,:])   
    def g(i,j) :                        ## T[i,j]
        return T[i,j]                   
    m = len(Xt)                         ## number of letters in the word
    b = get_bwd_msg(Xt,W,T)
    log_Zx = get_log_Zx(Xt,W,b)
    log_p_Yt = (f(m-1,Yt[m-1]) - log_Zx)
    for s in range(m-1) :
        log_p_Yt += f(s,Yt[s]) + g(Yt[s-1],Yt[s])   ## log(p(Yt|Xt)) = sum_s (f_s(y_s)) + sum_s(g(y_s-1,y_s) - log(Zx))
    return log_p_Yt

def get_grad_t(Yt,Xt,W,T) :             ## get log(p(Yt|Xt)) and gradients for one training example
    m = len(Xt)                         ## number of letters in the word
    a = get_fwd_msg(Xt,W,T)
    b = get_bwd_msg(Xt,W,T)
    log_Zx = get_log_Zx(Xt,W,b)
    
    def f(s,y) :                        ## < Wy, Xts >
        return np.dot(W[y,:],Xt[s,:])   
    def g(i,j) :                        ## T[i,j]
        return T[i,j]                   
    ## Calculate marginals
    def marginal_ys(s,ys) :
        return (np.exp(f(s,ys) + a[s,ys] + b[s,ys] - log_Zx))
    def marginal_ys_ys1(s,ys,ys1) :
        return (np.exp(f(s,ys) + f(s+1,ys1) + g(ys,ys1) + a[s,ys] + b[s+1,ys1] - log_Zx))

    ## Calculate Gradient wrt Wy of log(p(Y|X)) and collect them together
    grad_W_t = np.empty((26,128))
    for i in range(26) :
        res = np.zeros(128)
        for s in range(m) :
            ind = 1 if Yt[s] == i else 0
            res += (ind - marginal_ys(s,i)) * Xt[s,:]
        grad_W_t[i,:] = res

    ## Calculate Gradient wrt Tij of log(p(Y|X))
    grad_T_t = np.empty((26,26))
    for i in range(26) :
        for j in range(26) :
            res = 0
            for s in range(m-1) :
                ind = 1 if (Yt[s] == i and Yt[s+1] == j) else 0
                res += ind - marginal_ys_ys1(s,Yt[s],Yt[s+1])
            grad_T_t[i,j] = res
    
    ## flatten grad_W_t and grad_T_t and concatenate
    flat_grad_W_t = grad_W_t.flatten()
    flat_grad_T_t = grad_T_t.flatten('F')
    grad_theta_t = np.concatenate((flat_grad_W_t, flat_grad_T_t))                   ## Gradient vector
    
    return grad_theta_t


###################### Test code ######################

######################################################

def get_log_posterior(model,word_list) :
    W,T = extract_parameters(model)
    Y,X = extract_words(word_list)
    N = len(Y)                              ## Total number of words in the training data
    log_posterior = 0
    for t in range(N) :
        log_p_Yt = get_log_posterior_Yt(Y[t],X[t],W,T)
        log_posterior += log_p_Yt / N
    return log_posterior

def get_grad_forall(model,word_list) :
    W,T = extract_parameters(model)
    Y,X = extract_words(word_list)
    N = len(Y)                              ## Total number of words in the training data
    ## Calculate gradient of whole training data
    grad_theta = np.zeros(26*128 + 26*26)
    for t in range(N) :
        grad_theta_t = get_grad_t(Y[t], X[t], W, T)
        grad_theta += grad_theta_t / N
    return grad_theta

def write_grad_to_file(model,word_list) :
    grad_theta = get_grad_forall(model,word_list)

    ## Write gradient to file result/gradient.txt
    np.savetxt("../result/gradient.txt", grad_theta, fmt = '%f')


def crf_obj(model,word_list,C) : 
    log_posterior = get_log_posterior(model,word_list)
    obj = (np.sum(model**2) /2) - (C * log_posterior)     ## Objective function 
    return obj
def crf_grad(model,word_list,C) :
    grad_theta = get_grad_forall(model,word_list)
    grad_theta = model - (C * grad_theta)           ## Gradient for objective function
    return grad_theta


###################### Test code ######################

# use check_grad to check gradient value 

######################################################

def crf_test(model, word_list) :
    W,T = extract_parameters(model)
    Y,X = extract_words(word_list)
    
    y_predict = []
    for Xt in X :
        y_predict.append(decoder(Xt, W, T))                 ## Decode utility defined in 1c
    ## Make sure decoder returns only the y_predict and nothing else

    flat_Y = [ys for Yt in Y for ys in Yt]
    flat_y_predict = [ys for Yt in y_predict for ys in Yt]
    
    ## Calculate word-wise error
    num_words = len(Y)
    wordwise_error = 0
    for t in range(len(Y)) :
        err = np.count_nonzero(np.array(Y[t]) - np.array(y_predict[t]))
        if (err != 0) :
            wordwise_error += 1
    wordwise_error = wordwise_error * 100 / num_words

    ## Calculate letter-wise error
    num_letters = len(flat_Y)
    letterwise_error = np.count_nonzero(np.array(flat_Y) - np.array(flat_y_predict))
    letterwise_error = letterwise_error * 100 / num_letters

    return letterwise_error, wordwise_error, y_predict

def optimize_obj(train_data, test_data, C) :
    Wo = np.zeros((128*26+26**2,1))
    result = opt.fmin_tnc(crf_obj, Wo, args = [train_data, C], maxfun=100,
                          ftol=1e-3, disp=5)
    model = result[0]
    ## Store optimal solution W and T in result/solution.txt
    np.savetxt("../result/solution.txt", model, fmt = '%f')
    
    letterwise_error, wordwise_error, y_predict = crf_test(model, test_data)

    ## Store predictions for test data in result/prediction.txt
    np.savetxt("../result/prediction.txt", y_predict, fmt = '%i')
    
    # print('CRF test accuracy for c = {}: {}'.format(C,accuracy))
    return letterwise_error, wordwise_error