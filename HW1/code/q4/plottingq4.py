import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
x= ['0','500','1000','1500','2000']
y1 = [69.917,69.5,69.028,68.585,68.157]
y2 = [83,82,81,80,79]

w = 10
h = 8
d = 100
plt.figure(figsize=(w, h), dpi=d)

plt.plot(x, y1,marker='', color='olive', linewidth=2, linestyle='dashed', label="SVM_MC")
plt.plot(x, y2, color='green', linewidth=3,  label="CRF")
plt.legend()

plt.savefig("letteraccuracies.png")

y1 = [17.2,16.75,16.3,15.3,15.09] 
y2 = [40,38,36,34,32]

plt.figure(figsize=(w, h), dpi=d)

plt.plot(x, y1,marker='', color='olive', linewidth=2, linestyle='dashed', label="SVM_MC")
plt.plot(x, y2, color='green', linewidth=3,  label="CRF")
plt.legend()

plt.savefig("wordaccuracies.png")


