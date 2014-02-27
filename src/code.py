import numpy as np
from tens2mat import tens2mat
from kr import kr
from math import isinf
from mpcg import mpcg
from numpy import Inf, isnan, sqrt, ceil, log2, exp
from numpy.linalg.linalg import norm

'''
Numpy arrays definieren werkt met laatste index eerst verhogen, eerst in index laatst verhogen,
dus omgekeerd aan vectorisatie

Dus

T = np.array(    [[[111,112,113,114],
                [121,122,123,124],
                [131,132,133,134]],
                
                [[211,212,213,214],
                [221,222,223,224],
                [231,232,233,234]]])
T_ijk heeft als waarde i*100 + j*10 + k
                
Heeft als laatste element T[1,2,3] en is gelijk aan 234 (0-based indexering)
T.shape is gelijk aan (2,3,4)
'''

def getNbOfDimensions(tensor):
    return tensor.ndim
    
def getRank(U):
    return U[0].shape[1]

def getM(U, T):
    M = []
    for n in range(getNbOfDimensions(T)):
        M.append(tens2mat(T, n))
    return M
    
def copyListOfArray(U0):
    U = []
    for i in range(len(U0)):
        U.append(U0[i].copy())
    
    return U

def calculateOffset(size_tens, R):
    return np.hstack((np.array([0]), np.cumsum(np.array(size_tens)))) * R

def getDimensions(tensor):
    return tensor.shape
    
def cpd_nls(T,U0,options):
    
    # Check the initial factor matrices U0.
    N = getNbOfDimensions(T)
    
    U = copyListOfArray(U0)
    R = getRank(U0)

    size_tens = getDimensions(T)
    
    for e in U:
        if e.shape[1] != R:
            raise RuntimeError('cpd_nls:U0 size(U0{n},2) should be the same for all n.')
    
    for i in range(len(U)):
        if U[i].shape[0] != size_tens[i]:
            raise RuntimeError('cpd_nls:U0 size(T,n) should equal size(U0{n},1).')


    CGMaxIter = 10
    Display = 0
    JHasFullRank = False
    LargeScale = True
    #M = 'block-Jacobi'
    
    
    # Check the options structure.
    CGTol = 1e-6
    MaxIter = 200
    TolFun = 1e-12
    TolX = 1e-6
    Delta = NaN
    delta = Delta

    # Cache some intermediate variables.
    M = getM(U, T)
    
    offset = np.hstack((np.array([0]), np.array(size_tens))) * R
    
    UHU = calculateUHU(U, N, R)
    
    T2 = T.flatten('F').T.dot(T.flatten('F'))
    

    '''
    TODO 
    % Call the optimization method.
    dF.JHJx = @JHJx
    dF.JHF = @g;
    dF.M = @M_blockJacobi;
    
    [U,output] = options.Algorithm(@f,dF,U,options);
    output.Name = func2str(options.Algorithm);
    /TODO
    '''
    z0 = U    
    

    # Evaluate the function value at z0.
    dim = structure(z0)
    z = z0.copy()
    z0 = serialize(z0)
    
    fval = f(z)
    fval1 = fval
    
    '''
    
    % Gauss-Newton with dogleg trust region.
    output.alpha = [];
    output.cgiterations = [];
    output.cgrelres = [];
    output.delta = options.Delta;
    output.fval = fval;
    output.info = false;
    output.infops = [];
    output.iterations = 0;
    output.relfval = [];
    output.relstep = [];
    output.rho = [];
    '''
    
    info = False
    iterations = 0
    
    while not info:
        UHU = calculateUHU(U, N, R)
        grad = serialize(g(U, UHU, N, M))
        gg = grad.T.dot(grad)
        gBg = grad.T.dot(JHJx(z, UHU, N, R, offset, size_tens, grad))
        alpha = gg/gBg
        
        if not isinf(alpha):
            alpha = 1
        
        def JHJxs(x):
            JHJx(U, UHU, N, R, offset, size_tens, x)
        
        def PC(x):
            M_blockJacobi(x, N, UHU, offset, size_tens, R)
            
        pgn = mpcg(JHJxs, -1 * grad, PC, -alpha * grad, CGTol, CGMaxIter)
        
        rho = -Inf
        
        normpgn = norm(pgn)
        
        if isnan(delta):
            delta = max(1, normpgn)
        
        while rho <= 0:
            if normpgn <= delta:
                p = pgn
                dfval = -0.5*grad.T.dot(pgn)
            elif alpha*sqrt(gg) >= delta:
                p = (-delta/sqrt(gg))*grad
                dfval = delta*(sqrt(gg)-0.5*delta/alpha)
            else:
                bma = pgn+alpha*grad
                bmabma = bma.T.dot(bma)
                a = -alpha*grad
                aa = alpha*alpha*gg
                c = a.T.dot(bma)
                
                if(c <= 0):
                    beta = (-c + sqrt(c*c + bmabma*(delta*delta-aa)))/bmabma
                else:
                    beta = (delta*delta-aa)/(c+sqrt(c*c+bmabma*(delta*delta-aa)))
                
                p = a+beta*bma
                dfval = 0.5*alpha*(1-beta)*(1-beta)*gg - 0.5*beta*(2-beta)*(grad.T.dot(pgn))
            
            if dfval > 0:
                z = deserialize(z0 + p, dim)
                
                fvalO = fval
                fval = f(z)
                
                rho = (fvalO - fval)/dfval
                if isnan(rho):
                    rho = -Inf
                
            if rho > 0.5:
                delta = max(delta, 2*norm(p))
            else:
                sigma = (1-0.25)/(1+np.exp(-14*(rho-0.25)))+0.25
                if normpgn < sigma*delta and rho < 0:
                    e = ceil(log2(normpgn/delta)/log2(sigma))
                    delta = sigma^(exp(1)*delta)
                else:
                    delta = sigma * delta
            
            #Check for convergence.
            relstep = norm(p)/norm(z0)
            
            if isnan(relstep):
                relstep = 0
                
            if rho <= 0 and relstep <= TolX:
                z = deserialize(z0, dim)
                break
         
        if rho > 0:
            z0 = serialize(z)
        
        iterations = iterations + 1
        relfval = abs((fval - fvalO)/fval1)
        
        if relfval <= TolFun:
            info = 1
        
        if relstep <= TolX:
            info = 2
            
        if iterations >= MaxIter:
            info = 3

def deserialize(z, dim):
    r = []
    #s = cellfun(@(s)prod(s(:)),dim(:)); o = [0; cumsum(s)];
    s = []
    for i in range(len(dim)):
        s.append(np.prod(dim[i]))
    
    o = np.hstack((0, np.cumsum(np.array(s), axis = 0)))
    for i in range(len(s)):
        elements = o[i] + np.array(range(s[i]))
        r.append(z[elements].reshape(dim[i], order = 'F'))
    
    return r

def serialize(z):
    s = []
    for i in range(len(z)):
        s.append(z[i].size)
    
    o = np.hstack((0, np.cumsum(np.array(s), axis = 0)))
    r = np.zeros((o[-1]))

    for i in range(len(s)):
        elements = o[i] + np.array(range(s[i]))
        r[elements] = z[i].flatten(order = 'F')
        
    return r
    
def structure(z):
    dim = []
    for i in range(len(z)):
        dim.append(z[i].shape)
    
    return dim

def M_blockJacobi(b, N, UHU, offset, size_tens, R):    
    # Solve Mx = b, where M is a block-diagonal approximation for JHJ.
    # Equivalent to simultaneous ALS updates for each of the factor matrices.
    x = np.zeros(b.shape)
    r = np.array(range(N))

    for n in range(N):
        allButN = np.hstack((r[:n], r[n+1:N]))
        Wn = np.prod(UHU[:,:,allButN],axis = 2)
        
        idx = range(offset[n], offset[n+1])

        #A/B = (B'\A')'
        A = b[idx].copy().reshape((size_tens[n],R), order = 'F')
        x[idx] = np.linalg.solve(Wn.T, A.T).T.reshape((len(idx)), order = 'F')
    
    return x

def calculateUHU(U, N, R):
    '''
    calculates UHU
    
    @param U: U
    @param N: The number of dimensions
    @param R: The rank of the decomposition
    
    @return: (Unew, Uold, UHU)
    '''
    
    
    # Cache the Gramians U{n}'*U{n}.
    UHU = np.zeros((R,R,N))
    for n in range(N):
        UHU[:,:,n] = U[n].T.dot(U[n])

    return UHU

def f(U, M):
    D = M[0] - U[0].dot(kr(U[:0:-1]).T)
    fval = 0.5 * np.sum(D*D)
    return fval

def g(U, UHU, N, M):
    r = np.array(range(N))
    grad = [];
    
    for n in r:
        allButN = np.hstack((r[:n], r[n+1:N]))
        
        UallButNRev = []
        for i in reversed(allButN):
            UallButNRev .append(U[i])
        
        G1 = U[n].dot(np.prod(UHU[:,:,allButN], axis = 2))
        G2 = M[n].dot(kr(UallButNRev ))
        grad.append(G1-G2)
        
    return grad

def JHJx(U, UHU, N, R, offset, size_tens, x):
    # Compute JHJ*x.
    XHU = np.zeros(UHU.shape);
    y = np.zeros(x.shape);
    r = np.array(range(N))
    
    for n in range(N):
        allButN = np.hstack((r[:n], r[n+1:N]))

        idx = range(offset[n], offset[n+1])
        Wn = np.prod(UHU[:,:,allButN],axis = 2)
        Xn = x[idx].copy().reshape((size_tens[n],R), order = 'F')
        XHU[:,:,n] = Xn.T.dot(U[n])
        y[idx] = Xn.dot(Wn).reshape((len(idx)), order='F')
    
    for n in range(N-1):
        idxn = range(offset[n], offset[n+1])
        Wn = np.zeros(R)
        
        for m in range(n+1, N):
            allButNAndM = np.hstack((r[:n], r[n+1:m], r[m+1:N]))
            idxm = range(offset[m], offset[m+1])
            Wnm = np.prod(UHU[:,:,allButNAndM], axis = 2)
            Wn = Wn+Wnm*XHU[:,:,m]
            JHJmnx = U[m].dot(Wnm*XHU[:,:,n])
            y[idxm] = y[idxm]+JHJmnx.flatten(order = 'F')
        
        JHJnx = U[n].dot(Wn)
        y[idxn] = y[idxn]+JHJnx.flatten(order = 'F')
    return y