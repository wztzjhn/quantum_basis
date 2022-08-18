#!/usr/bin/python3

import sys
num_argv = len(sys.argv)

print('----------- generating triangular lattice toml file ----------')
if ( num_argv < 2 ):
    sys.exit("Usage: gen_triangular R")
elif ( sys.argv[1]=='--help' ):
    sys.exit("Usage: gen_triangular R")
else:
    R = int(sys.argv[1])
    print("R=",R)

N = 3*(R*R+R)+1
print("N=",N)

writefile = open("triangular_R"+str(R)+".toml", "w")
writefile.write("# Triangular cluster, with N="+str(N)+" sites (R="+str(R)+")\n")
writefile.write("# Note: N=3(R^2+R)+1\n")
writefile.write("\n")

writefile.write("dim = 2\n")
writefile.write("\n")
writefile.write("a0 = [ 1.0, 0.0 ]\n")
writefile.write("a1 = [ -0.5, 0.86602540378443865 ]\n")
writefile.write("\n")
writefile.write("A0 = ["+str(2*R+1)+", "+str(R)+" ]\n")
writefile.write("A1 = ["+str(R+1)+", "+str(2*R+1)+" ]\n")
writefile.write("\n")
writefile.write("num_sub = 1\n")
writefile.write("pos_sub0 = [ 0.0, 0.0 ]\n")
writefile.write("\n")

cnt=0
for y in range(-R,0):
    for x in range(-R,R+1+y):
        writefile.write("[[sub0]]\n")
        writefile.write("site = [ "+str(x)+", "+str(y)+" ]\n")
        cnt = cnt + 1
for y in range(0,R+1):
    for x in range(-R+y,R+1):
        writefile.write("[[sub0]]\n")
        writefile.write("site = [ "+str(x)+", "+str(y)+" ]\n")
        cnt = cnt + 1
assert(cnt == N)

writefile.close()
print('--------------------------------------------------------------')
