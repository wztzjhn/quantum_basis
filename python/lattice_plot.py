#!/usr/bin/python3
# a python code to plot the tilted lattice for double checking purpose
# first just write a 2D lattice plot, generalize later
import toml
import sys
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

num_argv = len(sys.argv)

if ( num_argv < 2 ):
    sys.exit("Usage: plot_lattice filename")
parsed_toml = toml.load(sys.argv[1])

dim = parsed_toml["dim"]
num_sub = parsed_toml["num_sub"]

if (dim != 2):
    sys.exit("currently only for plotting 2D lattices")

a = []
for d in range(0,dim):
    a.append(np.array(parsed_toml["a"+str(d)]))
a = np.array(a)

R = []
for d in range(0,dim):
    R.append(np.array(parsed_toml["A"+str(d)]))
R = np.array(R)

pos_sub = []
for sub in range(0,num_sub):
    pos_sub.append(np.array(parsed_toml["pos_sub"+str(sub)]))
pos_sub = np.array(pos_sub)

sites = []
carts = []
for sub in range(0,num_sub):
    site_temp = []
    cart_temp = []
    info = parsed_toml["sub"+str(sub)]
    for i in range(0,len(info)):
        site_temp.append(info[i]["site"])
        pos = ((info[i]["site"][0]+pos_sub[sub][0])*a[0]
              +(info[i]["site"][1]+pos_sub[sub][1])*a[1])
        cart_temp.append(pos)
    sites.append(np.array(site_temp))
    carts.append(np.array(cart_temp))

xmin = np.amin(carts[0][:,0])
xmax = np.amax(carts[0][:,0])
ymin = np.amin(carts[0][:,1])
ymax = np.amax(carts[0][:,1])
for sub in range(1,num_sub):
    if np.amin(carts[sub][:,0]) < xmin:
        xmin = np.amin(carts[sub][:,0])
    if np.amax(carts[sub][:,0]) > xmax:
        xmas = np.amax(carts[sub][:,0])
    if np.amin(carts[sub][:,1]) < ymin:
        ymin = np.amin(carts[sub][:,1])
    if np.amax(carts[sub][:,1]) > ymax:
        ymax = np.amax(carts[sub][:,1])


plotstyle = [ "o", "s", "^", "v", "*" ]
plotcolor = [ "r", "b", "m", "g", "c"]
for sub in range(0,num_sub):
    plt.plot(carts[sub][:,0],carts[sub][:,1],plotstyle[sub],color=plotcolor[sub])

for sub in range(0,num_sub):
    plt.plot(carts[sub][:,0]+(R[0][0]*a[0]+R[0][1]*a[1])[0],
             carts[sub][:,1]+(R[0][0]*a[0]+R[0][1]*a[1])[1],plotstyle[sub],color='orange')
    plt.plot(carts[sub][:,0]+(R[1][0]*a[0]+R[1][1]*a[1])[0],
             carts[sub][:,1]+(R[1][0]*a[0]+R[1][1]*a[1])[1],plotstyle[sub],color='grey')

#plt.xlim(math.floor(xmin-0.7),math.ceil(xmax+0.7))
#plt.ylim(math.floor(ymin-0.7),math.ceil(ymax+0.7))
plt.axes().set_aspect('equal')

plt.arrow(0,0,a[0][0],a[0][1],color='black')
plt.arrow(0,0,a[1][0],a[1][1],color='black')
plt.arrow(a[0][0],a[0][1],a[1][0],a[1][1],color='black')
plt.arrow(a[1][0],a[1][1],a[0][0],a[0][1],color='black')

plt.arrow(0,0,(R[0][0]*a[0]+R[0][1]*a[1])[0],(R[0][0]*a[0]+R[0][1]*a[1])[1],color='orange')
plt.arrow(0,0,(R[1][0]*a[0]+R[1][1]*a[1])[0],(R[1][0]*a[0]+R[1][1]*a[1])[1],color='grey')

plt.title("total sites: "+str(len(sites[0])*num_sub))

plt.show()
