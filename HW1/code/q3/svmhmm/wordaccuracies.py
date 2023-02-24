
lindex = {}
alphabets= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
digits = {'0':0,'1':1}
for i in range(len(alphabets)):
        lindex[alphabets[i]] = i+1
def returnlistoflabels(filename):
        newlist=[]
        fl = open(filename)
        for l in fl:
                l= l.strip()
                newlist.append(int(l))
        return newlist
        


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
        print(wordaccuracy)

temp = returnlistoflabels('p_labels100.txt')
for filename in ['p_labels1.txt','p_labels10.txt','p_labels100.txt','p_labels1000.txt','p_labels5000.txt']:
        print("Word accuracy for ",filename)
        wordaccuracy(returnlistoflabels(filename))





        



        





