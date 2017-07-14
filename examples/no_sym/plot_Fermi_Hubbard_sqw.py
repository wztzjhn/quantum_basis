#!/usr/bin/python
import sys
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

num_argv = len(sys.argv)

if ( num_argv<6 ):
    sys.exit("Usage: plot_sqw filename step eta omega_max scale_intensity ...")
elif ( sys.argv[1]=='--help' ):
    sys.exit("Usage: plot_sqw filename step eta omega_max scale_intensity...")
else:
    filename=sys.argv[1]
    steps   = int(sys.argv[2])
    broaden = float(sys.argv[3])
    omega_max = float(sys.argv[4])
    scale_intensity = float(sys.argv[5])


readfile = open(filename, "r")
print "Name of file to be processed: ", readfile.name
contents = readfile.readlines()
readfile.close()



Q = []
omega = []
nrm2_z = []
a_z = []
b_z = []

for line in range(0,len(contents)):
    line_str=contents[line].split('\n')[0]
    line_data=line_str.split()
    if len(line_data) == 0:
        continue
    elif line_data[0] == 'Lx':
        Lx = int(line_data[1])
        print 'Lx     =',Lx
    elif line_data[0] == 'Ly':
        Ly = int(line_data[1])
        print 'Ly     =',Ly
    elif line_data[0] == 'U':
        U = float(line_data[1])
        print 'U      =',U
    elif line_data[0] == 'E0':
        E0 = float(line_data[1])
        print 'E0    =',E0
    elif line_data[0] == 'Gap':
        Gap = float(line_data[1])
        print 'Gap   =',Gap
    elif line_data[0] == 'Q':
        q_tmp = line_data[1].split(',')
        for j in range(0,len(q_tmp)):
            q_tmp[j] = int(q_tmp[j])
        Q.append(q_tmp)
    elif line_data[0] == 'nrm2':
        nrm2_z.append(float(line_data[1]))
    elif line_data[0] == 'a':
        temp = []
        for i in range(1,len(line_data)):
            temp.append(float(line_data[i]))
        a_z.append(np.array(temp))
    elif line_data[0] == 'b':
        temp = []
        for i in range(1,len(line_data)):
            temp.append(float(line_data[i]))
        b_z.append(np.array(temp))

omega = np.linspace(0,omega_max,200+1)
Q_grid = np.arange(len(Q))
xx, yy = np.meshgrid(Q_grid,omega)
intensity_z = np.zeros((len(omega),len(Q)))
intensity_00   = np.zeros(len(omega))
intensity_hpi0 = np.zeros(len(omega)) #(pi/2, 0)
intensity_pi0  = np.zeros(len(omega))
intensity_pihpi = np.zeros(len(omega)) #(pi, pi/2)
intensity_pipi = np.zeros(len(omega))
intensity_hpihpi = np.zeros(len(omega)) #(pi/2, pi/2)


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
    return -2.0 * (nrm2*nrm2 / denorminator).imag

intensity_max = 0
for q_idx in range(0,len(Q)):
    momentum=Q[q_idx]
    for w_idx in range(0,len(omega)):
        if (nrm2_z[q_idx] > 0.000000001):
            temp = sqw(omega[w_idx],broaden,a_z[q_idx],b_z[q_idx],steps,nrm2_z[q_idx])
            if temp > intensity_max:
                intensity_max = temp
            intensity_z[w_idx][q_idx] = temp
        else:
            intensity_z[w_idx][q_idx] = 0.0
    if (momentum[0] == 0 and momentum[1] == 0):
        if (nrm2_z[q_idx] > 0.000000001):
            for w_idx in range(0,len(omega)):
                intensity_00[w_idx] = sqw(omega[w_idx],broaden,a_z[q_idx],b_z[q_idx],steps,nrm2_z[q_idx])
    elif (momentum[0] * 4 == Lx and momentum[1] == 0):
        if (nrm2_z[q_idx] > 0.000000001):
            for w_idx in range(0,len(omega)):
                intensity_hpi0[w_idx] = sqw(omega[w_idx],broaden,a_z[q_idx],b_z[q_idx],steps,nrm2_z[q_idx])
    elif (momentum[0] * 2 == Lx and momentum[1] == 0):
        if (nrm2_z[q_idx] > 0.000000001):
            for w_idx in range(0,len(omega)):
                intensity_pi0[w_idx] = sqw(omega[w_idx],broaden,a_z[q_idx],b_z[q_idx],steps,nrm2_z[q_idx])
    elif (momentum[0] * 2 == Lx and momentum[1] * 4 == Ly):
        if (nrm2_z[q_idx] > 0.000000001):
            for w_idx in range(0,len(omega)):
                intensity_pihpi[w_idx] = sqw(omega[w_idx],broaden,a_z[q_idx],b_z[q_idx],steps,nrm2_z[q_idx])
    elif (momentum[0] * 2 == Lx and momentum[1] * 2 == Ly):
        if (nrm2_z[q_idx] > 0.000000001):
            for w_idx in range(0,len(omega)):
                intensity_pipi[w_idx] = sqw(omega[w_idx],broaden,a_z[q_idx],b_z[q_idx],steps,nrm2_z[q_idx])
    elif (momentum[0] * 4 == Lx and momentum[1] * 4 == Ly):
        if (nrm2_z[q_idx] > 0.000000001):
            for w_idx in range(0,len(omega)):
                intensity_hpihpi[w_idx] = sqw(omega[w_idx],broaden,a_z[q_idx],b_z[q_idx],steps,nrm2_z[q_idx])

fig, ax = plt.subplots(1, 1, sharey=True)

max_color = float(str(scale_intensity*intensity_max)[:5])
levels = np.linspace(0,max_color,51)
cmap = plt.cm.get_cmap("jet")


cs0 = ax.contourf(xx,yy,intensity_z,levels,cmap=cmap,vmin=0.0,vmax=max_color)
fig.colorbar(cs0, ax=ax)
ax.set_xlabel('$Q$')

#cs1 = ax[1].contourf(xx,yy,intensity_y,levels,cmap=cmap,vmin=0.0,vmax=max_color)
#fig.colorbar(cs1, ax=ax[1])
#ax[1].set_xlabel('$Q$')
labels = ["$(0,0)$", "$(\pi/2,0)$", "$(\pi,0)$",
          "$(\pi,\pi/2)$", "$(\pi,\pi)$", "$(\pi/2,\pi/2)$", "$(0,0)$"]
ax.set_xticklabels(labels)
ax.set_ylabel('$\omega$')
ax.set_title('$S^{zz}$')
#ax[1].set_title('$S^{yy}$')
#ax[2].set_title('$S^{zz}$')
#ax[3].set_title('total')
plt.suptitle('Lx='+str(Lx)+', Ly='+str(Ly)+', U='+str(U) + ', E0='+str(E0)+
             ', Gap='+str(Gap)+', broaden='+str(broaden)+', Lanczos step = '+str(steps))
#plt.colorbar()

plt.figure(2)
plt.plot(omega,intensity_00,label="$Q=(0,0)$")
plt.plot(omega,intensity_pi0,label="$Q=(\pi,0)$")
plt.plot(omega,intensity_pipi,label="$Q=(\pi,\pi)$")
plt.legend()
plt.xlim(0,10)
plt.ylim(0,1.5)
plt.xlabel('$\omega$')
plt.ylabel("intensity")

plt.figure(3)
plt.plot(omega,intensity_hpi0,label="$Q=(\pi/2,0)$")
plt.plot(omega,intensity_pihpi,label="$Q=(\pi,\pi/2)$")
plt.plot(omega,intensity_hpihpi,label="$Q=(\pi/2,\pi/2)$")
plt.legend()
plt.xlim(0,10)
plt.ylim(0,1.5)
plt.xlabel('$\omega$')
plt.ylabel("intensity")


writefile = open("sqw_plot.dat", "w")
writefile.write("#(1)\t")
for i in range(2,7):
    writefile.write("(" + str(i) + ")" + "\t")
writefile.write("(" + str(i+1) + ")" + "\n")
writefile.write("omega\t")
writefile.write("(0,0)\t")
writefile.write("(pi/2,0)\t")
writefile.write("(pi,0)\t")
writefile.write("(pi,pi/2)\t")
writefile.write("(pi,pi)\t")
writefile.write("(pi/2,pi/2)\n")
for omega_label in range(0,len(omega)):
    omega_val = omega[omega_label]
    writefile.write(str(omega_val)+"\t")
    writefile.write(str(intensity_00[omega_label])+"\t")
    writefile.write(str(intensity_hpi0[omega_label])+"\t")
    writefile.write(str(intensity_pi0[omega_label])+"\t")
    writefile.write(str(intensity_pihpi[omega_label])+"\t")
    writefile.write(str(intensity_pipi[omega_label])+"\t")
    writefile.write(str(intensity_hpihpi[omega_label])+"\n")
writefile.close()

plt.figure(4)
plt.plot(omega,intensity_00,label="$Q=(0,0)$")
plt.plot(omega,intensity_hpi0,label="$Q=(\pi/2,0)$")
plt.plot(omega,intensity_pi0,label="$Q=(\pi,0)$")
plt.plot(omega,intensity_pihpi,label="$Q=(\pi,\pi/2)$")
plt.plot(omega,intensity_pipi,label="$Q=(\pi,\pi)$")
plt.plot(omega,intensity_hpihpi,label="$Q=(\pi/2,\pi/2)$")
plt.legend()
plt.xlim(0,10)
plt.xlabel('$\omega$')
plt.ylabel("intensity")

plt.show()
