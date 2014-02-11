import numpy as np
'''
function [x,flag,relres,iter] = mpcg(A,b,tol,maxit,M,~,x0)
%MPCG Modified preconditioned conjugate gradients method.
%   x = mpcg(A,b) attempts to solve the system of linear equations A*x = b
%   for x. The n-by-n coefficient matrix A must be symmetric and positive
%   definite. The column vector b must have length n. A can be a function
%   handle afun such that afun(x) returns A*x. MATLAB's pcg.m can in some
%   cases return x0 (which is often equal to 0) if it is the iterate with
%   the smallest residual. The only difference between mpcg and MATLAB's
%   pcg method is that this implementation returns the last iterate,
%   regardless of its residual.
%
%   mpcg(A,b,tol) specifies the tolerance of the method. If tol is [], then
%   mpcg uses the default, 1e-6.
%
%   mpcg(A,b,tol,maxit) specifies the maximum number of iterations. If
%   maxit is [], then mpcg uses the default, min(n,20).
%
%   mpcg(A,b,tol,maxit,M) uses a symmetric positive definite preconditioner
%   M and effectively solve the system inv(M)*A*x = inv(M)*b for x. If M is
%   [] then mpcg applies no preconditioner. M can be a function handle mfun
%   such that mfun(x) returns M\x. If M is 'SSOR', then mpcg applies a
%   Symmetric Successive Over-Relaxation preconditioner.
%
%   mpcg(A,b,tol,maxit,M,[],x0) specifies the initial guess. If x0 is [],
%   then mpcg uses the default, an all-zero vector.
%
%   [x,flag,relres,iter] = mpcg(A,b,...) also returns a convergence flag,
%   which is 0 if mpcg converged to the desired tolerance within maxit
%   iterations and 1 otherwise, the relative residual norm(A*x-b)/norm(b)
%   and the number of iterations, respectively.
%
%   See also bicgstab, cgs, gmres, lsqsr, qmr.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] J. Nocedal and S. J. Wright, "Numerical Optimization," Springer
%       Series in Operations Research, Springer, second ed., 2006.

'''

def mpcg(A,b,M,x0,tol = 1e-6, maxit = 20):
    maxit = min((maxit, b.size))

    '''    
    A is een functie, niet numeriek.
    Idem voor M
    
    TODO: algemeen: dimensie-check
    
    '''
    x = x0.copy()
    r = A(x)-b
    y = M(r)
    d = -y
    rr = r.T.dot(y)
    
    normb = np.linalg.norm(b)
    
    for iter in range(maxit):
        Ad = A(d)
        #TODO check dimensions
        #alpha = rr/(d'*Ad);
        alpha = rr/(d.T.dot(Ad))
        
        x = x+alpha*d
        r = r+alpha*Ad
        rr1 = rr.copy()
        y = M(r)
        rr = r.T.dot(y)
        
        relres = np.linalg.norm(r)/normb
        if(relres < tol):
            break
        
        beta = rr/rr1
        d = -y+beta*d
        
    return x