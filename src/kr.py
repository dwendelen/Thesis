import numpy as np

#function X = kr(U,varargin)
def kr(U):
    '''
    Matlab doc:
    %KR Khatri-Rao product.
    %   kr(A,B) returns the Khatri-Rao product of two matrices A and B, of 
    %   dimensions I-by-K and J-by-K respectively. The result is an I*J-by-K
    %   matrix formed by the matching columnwise Kronecker products, i.e.,
    %   the k-th column of the Khatri-Rao product is defined as
    %   kron(A(:,k),B(:,k)).
    %
    %   kr(A,B,C,...) and kr({A B C ...}) compute a string of Khatri-Rao 
    %   products A x B x C x ..., where x denotes the Khatri-Rao product.
    %
    %   See also kron.
    
    %   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
    %            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
    %            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
    '''

    (J, K) = U[-1].shape
    
    #X = U[-1].copy().reshape((J, 1, K), order = 'F')
    X = U[-1].reshape((J, 1, K), order = 'F')

    for n in range(len(U)-2, -1, -1):
        I = U[n].shape[0]
        A = U[n].reshape((1, I, K), order='F')
        X = (A*X).reshape((I*J, 1, K), order='F')
        J = I*J
        
    X = X.reshape((X.shape[0], K), order='F')
    
    return X