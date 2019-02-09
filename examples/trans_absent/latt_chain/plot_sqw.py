#!/usr/bin/python
import sys
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

num_argv = len(sys.argv)

if ( num_argv<6 ):
    sys.exit("Usage: plot_sqw filename step eta omega_max max_intensity ...")
elif ( sys.argv[1]=='--help' ):
    sys.exit("Usage: plot_sqw filename step eta omega_max max_intensity...")
else:
    filename=sys.argv[1]
    steps   = int(sys.argv[2])
    broaden = float(sys.argv[3])
    omega_max = float(sys.argv[4])
    max_intensity = float(sys.argv[5])


readfile = open(filename, "r")
print "Name of file to be processed: ", readfile.name
contents = readfile.readlines()
readfile.close()



Q = []
omega = []
nrm2 = []
a = []
b = []

for line in range(0,len(contents)):
    line_str=contents[line].split('\n')[0]
    line_data=line_str.split()
    if len(line_data) == 0:
        continue
    elif line_data[0] == 'L':
        L = int(line_data[1])
        print 'L     =',L
    elif line_data[0] == 'J1':
        J1 = float(line_data[1])
        print 'J1    =',J1
    elif line_data[0] == 'J2':
        J2 = float(line_data[1])
        print 'J2    =',J2
    elif line_data[0] == 'E0':
        E0 = float(line_data[1])
        print 'E0    =',E0
    elif line_data[0] == 'Gap':
        Gap = float(line_data[1])
        print 'Gap   =',Gap
    elif line_data[0] == 'Q':
        Q.append(float(line_data[1]))
    elif line_data[0] == 'nrm2':
        nrm2.append(float(line_data[1]))
    elif line_data[0] == 'a':
        temp = []
        for i in range(1,len(line_data)):
            temp.append(float(line_data[i]))
        a.append(np.array(temp))
    elif line_data[0] == 'b':
        temp = []
        for i in range(1,len(line_data)):
            temp.append(float(line_data[i]))
        b.append(np.array(temp))

omega = np.linspace(0,omega_max,100+1)
xx, yy = np.meshgrid(Q,omega)
intensity_total = np.zeros((len(omega),len(Q)))
intensity_pi = np.zeros(len(omega))


def cfraction(a,b,length):
    res = 0.0
    for j in range(length-1,0,-1):
        res = -b[j] * b[j] / (a[j] + res)
    return a[0] + res

def sqw(w,eta,a,b,length,nrm2):
    a2 = []
    #print "w + E0  = ", w + E0 + complex(0,eta)
    for j in range(0,length):
        a2.append(w + E0 + complex(0,eta) - a[j])
    denorminator = cfraction(a2,b,length)
    return -2.0 * (nrm2*nrm2 / np.pi / denorminator).imag

intensity_max = 0
for q_idx in range(0,len(Q)):
    for w_idx in range(0,len(omega)):
        if (nrm2[q_idx] > 0.000000001):
            temp = sqw(omega[w_idx],broaden,a[q_idx],b[q_idx],steps,nrm2[q_idx])
            if temp > intensity_max:
                intensity_max = temp
            intensity_total[w_idx][q_idx] = temp
        else:
            intensity_total[w_idx][q_idx] = 0.0
    if (abs(Q[q_idx]- 3.1415926) < 0.00001):
        for w_idx in range(0,len(omega)):
            if (nrm2[q_idx] > 0.000000001):
                intensity_pi[w_idx] = (intensity_pi[w_idx] +
                    sqw(omega[w_idx],broaden,a[q_idx],b[q_idx],steps,nrm2[q_idx]))

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(10, 7))

colors = ['#0000fe', '#00cbfe', '#00fe13', '#f2fe00', '#fe0900']  # in terms of rgb
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

max_color = max_intensity
#float(str(scale_intensity*intensity_max)[:4])
levels = np.linspace(0,max_color,101)
# cmap = plt.cm.get_cmap("jet")


ax.set_title('L='+str(L)+', J1='+str(J1)+', J2='+str(J2) + ', E0='+str(E0)+
             ', Gap='+str(Gap)+', broaden='+str(broaden)+', Lanczos step = '+str(steps))

ax.tick_params(direction='in')
ax.set_xlabel('$\mathregular{Q}$')
ax.set_ylabel('$\omega$')

ax.set_ylim([0, omega_max])
#ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
#ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8'])

cs0 = ax.contourf(xx,yy,intensity_total,levels,cmap=cmap,vmin=0.0,vmax=max_color,extend="max")
cbar = fig.colorbar(cs0, ax=ax, pad=0.02)
#cbar.set_ticks([0.0,0.1,0.2,0.3])
cbar.outline.set_visible(False)
cbar.ax.tick_params(length=0)
cbar.ax.set_ylabel('Int', rotation=90)

#plt.tight_layout()



plt.figure(2)
plt.plot(omega,intensity_pi,label="$Q=\pi$")
plt.legend()
plt.xlabel('$\omega/J$')
plt.ylabel("intensity")

plt.show()
