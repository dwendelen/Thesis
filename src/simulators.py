import math

import matplotlib.pyplot as plt
import numpy as np
from Kernel.Float3DElement import Float3DElement
from Kernel.Float16x16x16 import Float16x16x16

'''
@return: (tcalc, tmemm, tmemz)
'''
def simulateKernel(kernel, I, R, n, perBasicElement = False):
    tcalc = kernel.getNbOperaties(I, R, n)/2703.36e9
    bes = kernel.getBasicElements(I, R, n)
    
    tmemz = kernel.getDataTransferZonderCache(I, R, n)/176.0e9
    tmemm = kernel.getDataTransferMetCache(I, R, n)/176.0e9
    
    if(perBasicElement):
        return(tcalc/bes, tmemm/bes, tmemz/bes)
    else:
        return(tcalc, tmemm, tmemz)

def cpu(I, R):
    flops = (2 + 2*R)*I*I*I + R*I*I
    tcalc = flops/(2.4*4e9)
    
    tcalcPBasicElem = tcalc/(I*I*I*R)
    
    nbBytes = (I*I*I + 3*I*R) * 4
    tmem = nbBytes/(32e9)
    tmemPBasicElem = tmem/(I*I*I*R)
    
    return (tcalcPBasicElem, tmemPBasicElem, tcalc, tmem)


def simAndPlotKernels(kernels, start, length, R, n):
    
    x = range(start, start+length)

    plt.hold(True)
    for kernel in kernels:
        print kernel
        calc = np.zeros((length))
        memm = np.zeros((length))
        memz = np.zeros((length))
        
        for i in range(0, length):
            j = i + start
            (calc[i], memm[i], memz[i]) = simulateKernel(kernel, j, R, n, True)
            
        plt.plot(x, calc)
        plt.plot(x, memm)
        #plt.plot(x, memz)
        
    plt.show()
    plt.plot(False)

R=40    
simAndPlotKernels([Float3DElement(), Float16x16x16()], 10, 20, R, 3)