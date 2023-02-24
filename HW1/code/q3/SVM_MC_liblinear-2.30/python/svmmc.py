from scipy.sparse import csr_matrix
import numpy as np
from liblinearutil import *

lindex = {}
alphabets= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
digits = {'0':0,'1':1}
for i in range(len(alphabets)):
        lindex[alphabets[i]] = i+1

ftrain = open('train.txt')
xid,xletter,xnextid,xwordid,xpixels,xposition = [],[],[],[],[],[]
for line in ftrain:
    fields=line.split(" ")
    fields[-1] = fields[-1].strip()
    xid.append(fields[0])
    xletter.append(lindex[fields[1]])
    xnextid.append(fields[2])
    xwordid.append(fields[3])
    xposition.append(fields[4])
    temp=[]
    for i in range(len(fields[5:])):
        temp.append(digits[fields[5+i]])
    xpixels.append(temp)
ftrain.close()  

ftest = open('test.txt')
txid,txletter,txnextid,txwordid,txpixels,txposition = [],[],[],[],[],[]
for line in ftest:
    tfields=line.split(" ")
    tfields[-1] = tfields[-1].strip()
    txid.append(tfields[0])
    txletter.append(lindex[tfields[1]])
    txnextid.append(tfields[2])
    txwordid.append(tfields[3])
    txposition.append(tfields[4])
    temp=[]
    for i in range(len(tfields[5:])):
        temp.append(digits[tfields[5+i]])
    txpixels.append(temp)
ftest.close()


ftest.close()


with open('dftrain.txt', 'w') as filehandle:
    for li in range(len(xletter)):
        filehandle.write('%s ' % xletter[li])
        for i in range(len(xpixels[li])):
            filehandle.write('%s:' % (i+1) )
            filehandle.write('%s ' %xpixels[li][i] )
        filehandle.write('\n')

ytrain,xtrain=svm_read_problem('dftrain.txt',return_scipy = True)  


with open('dftest.txt', 'w') as filehandle:
    for li in range(len(txletter)):
        filehandle.write('%s ' % txletter[li])
        for i in range(len(txpixels[li])):
            filehandle.write('%s:' % (i+1) )
            filehandle.write('%s ' %txpixels[li][i] )
        filehandle.write('\n')

ytest,xtest=svm_read_problem('dftest.txt',return_scipy = True) 

#ytrain = np.asarray(xletter, dtype=np.int32)
#ytest = np.asarray(txletter, dtype=np.int32)
#xtrain = csr_matrix(xpixels)
#xtest = csr_matrix(txpixels)
#print(len(ytrain),len(xtrain),len(ytest),len(xtest))
#'-c 3.38e-5','-c 3.38e-4','-c 3.38e-3','-c 3.38e-2','-c 1.69e-1','-c 1.69e+2']'

def wordaccuracy(p_label):
        incorrectwords = 0
        totalwords = 0
        flag = True
        for i in range(len(p_label)):
                if p_label[i] != txletter[i]:
                        flag= False
                if txnextid[i] == '-1':
                        if flag == False:
                                incorrectwords+=1
                                flag  = True
                        totalwords = totalwords+1

        wordaccuracy = 1 - incorrectwords/totalwords
        return wordaccuracy
cvals = [1,10,100,1000,5000]
for cval in ['-c 3.38e-5','-c 3.38e-4','-c 3.38e-3','-c 3.38e-2','-c 1.69e-1','-c 1.69e+2']:
        print("Model with %s",cval)
        m = train(ytrain,xtrain,cval)
        p_label,p_acc,p_val = predict(ytest,xtest,m)
        print("Word accuracy: ", wordaccuracy(p_label))
        print("\n")
        





        



        





