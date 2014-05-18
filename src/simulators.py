import math

import matplotlib.pyplot as plt
import numpy as np

def float16x16x16SlechtsEenKeerFetchen(I, R):
    nbWGs = pow(math.ceil(I/16.0), 3)
    flops = nbWGs * (9216*R + 12287)
    tcalc = flops/(2703.36e9)
    
    tcalcPBasicElem = tcalc / (I*I*I*R)

    i = math.ceil(I/16.0) * 16
    nbBytes = (i*i*i + 3*i*R) * 4
    tmem = nbBytes/176.0e9
    tmemPBasicElem = tmem/(I*I*I*R)
    
    return (tcalcPBasicElem, tmemPBasicElem, tcalc, tmem)

def float16x16x16ElkeKeerNieuweFetch(I, R):
    nbWGs = pow(math.ceil(I/16.0), 3)
    flops = nbWGs * (9216*R + 12287)
    tcalc = flops/(2703.36e9)
    
    tcalcPBasicElem = tcalc / (I*I*I*R)

    nbBytes = 4*nbWGs * (16*16*16 + 3 * 16 * R)
    tmem = nbBytes/176.0e9
    tmemPBasicElem = tmem/(I*I*I*R)
    
    return (tcalcPBasicElem, tmemPBasicElem, tcalc, tmem)

def cpu(I, R):
    flops = (2 + 2*R)*I*I*I + R*I*I
    tcalc = flops/(2.4*4e9)
    
    tcalcPBasicElem = tcalc/(I*I*I*R)
    
    nbBytes = (I*I*I + 3*I*R) * 4
    tmem = nbBytes/(32e9)
    tmemPBasicElem = tmem/(I*I*I*R)
    
    return (tcalcPBasicElem, tmemPBasicElem, tcalc, tmem)

N = 100
offset = 10

calc1 = np.zeros((N))
mem1 = np.zeros((N))

calcCpu = np.zeros((N))
memCpu = np.zeros((N))

R = 26
for i in range(0, N):
    (calc1[i], mem1[i], _, _) = float16x16x16SlechtsEenKeerFetchen(i+offset, R)
    (calcCpu[i], memCpu[i], _, _) = cpu(i+offset, R)
    
calc2 = np.zeros((N))
mem2 = np.zeros((N))
R = 20
for i in range(0, N):
    (calc2[i], mem2[i], _, _) = float16x16x16ElkeKeerNieuweFetch(i+offset, R)

x = range(offset, N+offset)

plt.hold(True)
plt.plot(x, mem1, 'b')
plt.plot(x, calc1, 'r')

#plt.plot(x, memCpu, 'r')
#plt.plot(x, calcCpu, 'b')
plt.show()
plt.plot(False)