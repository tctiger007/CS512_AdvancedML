import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
x= ['1','10','100','1000','5000']
y1 = [67.56,75.33,82.23,84.62,85.24]
y2 = [48.33,61.17,68.05,69.72,69.95]
y3 = [30,40,60,70,80]

w = 10
h = 8
d = 100
plt.figure(figsize=(w, h), dpi=d)

plt.plot(x, y1,marker='', color='olive', linewidth=2, linestyle='dashed', label="SVM_HMM")
plt.plot(x, y2, color='red', linewidth=2, linestyle='dashed', label="SVM_MC")
plt.plot(x, y3, color='green', linewidth=3,  label="CRF")
plt.legend()

plt.savefig("letteraccuracies.png")

y1 = [16.66,26.22,41.3,48.2,49.4]
y2 = [2,7.64,14.94,16.92,17.09] 
y3 = [30,40,60,70,80]

plt.figure(figsize=(w, h), dpi=d)

plt.plot(x, y1,marker='', color='olive', linewidth=2, linestyle='dashed', label="SVM_HMM")
plt.plot(x, y2, color='red', linewidth=2, linestyle='dashed', label="SVM_MC")
plt.plot(x, y3, color='green', linewidth=3,  label="CRF")
plt.legend()

plt.savefig("wordaccuracies.png")


