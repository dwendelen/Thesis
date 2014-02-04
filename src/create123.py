import numpy as np
from code import cpd_nls

#Some testingdata
U=[]
U.append(np.array([[1,2]]))
U.append(np.array([[1,2],[3,4]]))
U.append(np.array([[1,2],[3,4],[5,6]]))

T = np.zeros((1,2,3));
T[0, 0, 0] = 111;
T[0, 1, 0] = 121;
T[0, 0, 1] = 112;
T[0, 1, 1] = 122;
T[0, 0, 2] = 113;
T[0, 1, 2] = 123;


cpd_nls(T, U, 0)