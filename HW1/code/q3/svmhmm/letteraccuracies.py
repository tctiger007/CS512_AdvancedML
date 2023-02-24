
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



def letteraccuracy(p_label):
        incorrectletters = 0
        totalletters= len(p_label)
        for i in range(totalletters):
                if p_label[i] != txletter[i]:
                        incorrectletters+=1                 

        letteraccuracy = 1 - incorrectletters/totalletters
        print(letteraccuracy)

temp = returnlistoflabels('p_labels100.txt')
for filename in ['p_labels1.txt','p_labels10.txt','p_labels100.txt','p_labels1000.txt','p_labels5000.txt']:
        print("letter accuracy for ",filename)
        letteraccuracy(returnlistoflabels(filename))





        



        





