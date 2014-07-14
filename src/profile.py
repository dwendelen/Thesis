import numpy as np
import cProfile
from code import cpd_nls
'''
I = 100
R = 50

T = np.random.rand(I,I,I)
U = np.random.rand(I, R)

cpd_nls(T, (U,U,U))

cProfile.run('cpd_nls(T, (U,U,U))')'''

f = np.array(np.random.rand(10000000), order = 'F')
c = np.array(np.random.rand(10000000), order = 'C')

cProfile.run('f.dot(c.T)')
cProfile.run('f.dot(f.T)')
cProfile.run('c.dot(c.T)')
cProfile.run('c.dot(f.T)')