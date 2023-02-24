import numpy as np
import scipy.optimize as opt
import string
import math

path = "../"

decode_input = np.genfromtxt(path + "data/decode_input.txt", delimiter = ' ')

X = decode_input[:100*128]
W = decode_input[100*128:100*128+26*128]
T = decode_input[100*128+26*128:]

W = np.reshape(W, (26, 128))  # each row of W is w_y (128 dim)
T = np.reshape(T, (26, 26))   # T is 26*26
T = T.transpose()             # To make T11, T21, T31 ... T26,1 the first items of the rows 
m = int(len(X)/128)           # length of the word 
X = np.reshape(X, (m, 128))

def f(s,i) : # Node potential for < Wys = i, X^t_s >
    return np.dot(W[i,:],X[s,:])  
def g(i,j) : # Edge potential for Tys =i, ys+1 = j
    return T[i,j] 

# here we calculates all l1(y1) to l_m-1(y_m-1) using equations 27 and 28
# be carefule that the last node does not take account the edge potential 
# and therefore does not involve g function (which is T)
l = np.zeros((m, 26)) # storing all l's where rows represents 1, 2 ,,, s, ... m letter images; and cols 1...26

def decoder(X, W, T): 
    for s in range(1, m) :        # s taking 1 to m-1; where s: sth letter in X^t matrix
        for j in range(26) :      # y_s+1 = j
            temp = []
            for i in range(26) :  # y_s = i
                temp.append(f(s-1,i) + g(i,j) + l[s-1,i]) # calculate equation 28 till l_m-1(y_m-1)
            l[s,j] = max(temp) 
    l_m = []                      # ym (equation 29).
    for i in range(26):      
        l_m = np.append(l_m, [f(m-1,i) + l[m-1,i]])
    # max_m = np.amax(l_m)              # max{<Wym, xm> + lm(ym)} gives a value 200.18515048829295
    print(np.amax(l_m))
    
    y_pred = np.zeros((m), dtype = int)
    y_pred[m-1] = np.argmax(l_m)     # ym^*

    # Now we are doing argmax 
    for s in range(m-2, -1, -1):
        temp1 = []
        for i in range(26):
            temp1.append(f(s, i) + g(i, y_pred[s+1]) + l[s, i])
        y_pred[s] = np.argmax(temp1)
    return y_pred + 1   # need to add 1 to the prediction labels so that 1 -> a, ... 26 -> z

# print(decoder(X, W, T)[0]) # report the maximum objective function 200.18515048829298

decode_output = decoder(X, W, T)

np.savetxt(path + "result/decode_output.txt", decode_output, fmt = '%i')






