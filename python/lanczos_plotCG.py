#!/usr/bin/python

# used for plotting Conjugate Gradient progress (accuracy)

from sys import argv
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

readfile = open(argv[1], "r")
data = [line.split() for line in readfile]
readfile.close()
filename = argv[1].split("/")[-1].split(".")[0]

# function used to check if the argument string can be converted into float
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

row_total=len(data)
row_end   = row_total-1
# remove comment lines and blank lines
for row in range(row_end,-1,-1):
    if ( not(data[row]) ):
        del data[row]
    elif ( data[row][0][0]=='#' or not(is_number(data[row][0])) ):
        del data[row]
row_total=len(data)
row_end   = row_total-1

x  = np.zeros(row_total)
y  = np.zeros(row_total)
cnt  = 0
cnt0 = 0
for row in range(0,row_total):
    cnt = cnt + 1
    if (abs(float(data[row][0])-1.0) < 0.00001):
        cnt0 = cnt0 + cnt
        cnt = 0
    x[row] = float(data[row][0]) + cnt0
    y[row] = float(data[row][1])

fig, ax = plt.subplots(1, figsize=(6,4))
ax.set_ylabel("Accuracy")
ax.set_yscale("log")
ax.set_ylim(1e-14,1)
ax.plot(x,y,"+-")
ax.plot(x,1e-12*np.ones(row_total),"k--")

plt.legend()
plt.tight_layout()
plt.savefig(filename+".png")
plt.show()
