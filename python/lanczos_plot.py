#!/usr/bin/python

# used for plotting Lanczos progress (energy and accuracy)

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
    elif ( data[row][0][0]=='#' or not(is_number(data[row][1])) ):
        del data[row]
row_total=len(data)
row_end   = row_total-1

x  = np.zeros(row_total)
r0 = np.zeros(row_total)
r1 = np.zeros(row_total)
r2 = np.zeros(row_total)
r3 = np.zeros(row_total)
accu  = np.zeros(row_total)
accu0 = np.zeros(row_total)
accu1 = np.zeros(row_total)
for row in range(0,row_total):
    x[row]  = float(data[row][0])
    r0[row] = float(data[row][1])
    r1[row] = float(data[row][2])
    r2[row] = float(data[row][3])
    r3[row] = float(data[row][4])
    accu[row]  = float(data[row][7])
    accu0[row] = float(data[row][8])
    accu1[row] = float(data[row][9])

fig, ax = plt.subplots(2, sharex=True, figsize=(6,8))

ymin = np.floor(r0[row_end-1])
ymax = ymin + 3
ax[0].set_ylim(ymin,ymax)
ax[0].set_ylabel("Ritz values")
ax[0].plot(x,r0,"+-")
ax[0].plot(x,r1,"x-")
ax[0].plot(x,r2,"s-")
ax[0].plot(x,r3,"o-")

ax[1].set_ylabel("Accuracy")
ax[1].set_yscale("log")
ax[1].set_ylim(1e-16,1)
ax[1].set_xlabel(r'$m$')
ax[1].plot(x,accu0,"+-", label=r'$|\Delta E_0 / E_0|$')
ax[1].plot(x,accu1,"x-", label=r'$|\Delta E_1 / E_1|$')
ax[1].plot(x,accu,"md-", label=r'$||H \phi_0 -E_0 \phi_0 ||$')
ax[1].plot(x,1e-12*np.ones(row_total),"k--")

plt.legend(loc=3)
plt.tight_layout()
plt.savefig(filename+".png")
plt.show()
