import numpy as np
from tens2mat import tens2mat
from kr import kr

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

def cpd_nls(T,U0,options):
    
    # Check the initial factor matrices U0.
    N = T.ndim
    
    U = []
    for i in range(len(U0)):
        U.append(U0[i].copy())
    
    R = U0[1].shape[1]

    size_tens = T.shape
    
    for e in U:
        if e.shape[1] != R:
            raise RuntimeError('cpd_nls:U0 size(U0{n},2) should be the same for all n.')
    
    for i in range(len(U)):
        if U[i].shape[0] != size_tens[i]:
            raise RuntimeError('cpd_nls:U0 size(T,n) should equal size(U0{n},1).')
    
    '''
    TODO

    CGMaxIter = 10
    Display = 0
    JHasFullRank = false
    LargeScale = true
    M = 'block-Jacobi'
    
    
    % Check the options structure.

    CGTol = 1e-6;
    MaxIter = 200;
    PlaneSearch = false;
    PlaneSearchOptions = struct;
    TolFun = 1e-12;
    TolX = 1e-6;
    Delta = nan;
    
    /TODO
    '''

    # Cache some intermediate variables.
    M = []
    for n in range(N):
        M.append(tens2mat(T, n))
    
    offset = np.hstack((np.array([0]), np.array(size_tens))) * R
    
    print "U1:"; print U
    UHU = np.array([])
    (U, U0, UHU) = updateUHU(U, U0, UHU, N, R)
    
    T2 = T.flatten('F').T.dot(T.flatten('F'))

    print "F:"; print f(U, M)
    

'''
TODO 
% Call the optimization method.
dF.JHJx = @JHJx
dF.JHF = @g;
dF.M = @M_blockJacobi;




[U,output] = options.Algorithm(@f,dF,U,options);
output.Name = func2str(options.Algorithm);
'''
    
def f(U, M):
    D = M[0] - (U[0].dot(kr(U[:0:-1]))).T
    #D = M{1}-U{1}*kr(U(end:-1:2)).';
    fval = 0.5 * np.sum(D*D)
    #fval = 0.5*sum(D(:)'*D(:));
    return fval

'''
function grad = g(U)
    updateUHU(U);
    grad = cell(1,N);
    for n = 1:N
        G1 = U{n}*conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        G2 = M{n}*conj(kr(U([N:-1:n+1 n-1:-1:1])));
        grad{n} = G1-G2;
    end
end


function y = JHJx(U,x)
% Compute JHJ*x.
    XHU = zeros(size(UHU));
    y = zeros(size(x));
    for n = 1:N
        idx = offset(n)+1:offset(n+1);
        Wn = conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        Xn = reshape(x(idx),size_tens(n),R);
        XHU(:,:,n) = Xn'*U{n};
        y(idx) = Xn*Wn;
    end
    for n = 1:N-1
        idxn = offset(n)+1:offset(n+1);
        Wn = zeros(R);
        for m = n+1:N
            idxm = offset(m)+1:offset(m+1);
            Wnm = conj(prod(UHU(:,:,[1:n-1 n+1:m-1 m+1:N]),3));
            Wn = Wn+Wnm.*conj(XHU(:,:,m));
            JHJmnx = U{m}*(Wnm.*conj(XHU(:,:,n)));
            y(idxm) = y(idxm)+JHJmnx(:);
        end
        JHJnx = U{n}*Wn;
        y(idxn) = y(idxn)+JHJnx(:);
    end
end

function x = M_blockJacobi(~,b)
% Solve Mx = b, where M is a block-diagonal approximation for JHJ.
% Equivalent to simultaneous ALS updates for each of the factor matrices.
    x = zeros(size(b));
    for n = 1:N
        Wn = conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        idx = offset(n)+1:offset(n+1);
        x(idx) = reshape(b(idx),size_tens(n),R)/Wn;
    end
end
'''
    

def updateUHU(Unew, Uold, UHU, N, R):
    '''
    Updates UHU
    
    @param Unew: The new U
    @param Uold: The old U
    @param UHU: UHU to update
    @param N: The number of dimensions
    @param R: The rank of the decomposition
    
    @return: (Unew, Uold, UHU)
    '''
    
    
    # Cache the Gramians U{n}'*U{n}.
    newState = (UHU.size == 0)
    for n in range(N):
        if newState:
            break
        
        newState = np.array_equal(Uold[n], Unew[n])
    
    if newState:        
        if UHU.size == 0:
            UHU = np.zeros((R,R,N))
        for n in range(N):
            UHU[:,:,n] = Unew[n].T.dot(Unew[n])

    return (Unew, Unew, UHU)

'''
end

function [z,output] = nls_gndl(F,dF,z0,options)
%NLS_GNDL Nonlinear least squares by Gauss-Newton with dogleg trust region.
%   [z,output] = nls_gndl(F,dF,z0) starts at z0 and attempts to find a
%   local minimizer of the real-valued function f(z), which is the
%   nonlinear least squares objective function f(z) := 0.5*(F(z)'*F(z)).
%   The input variable z may be a scalar, vector, matrix, tensor or even a
%   cell array of tensors and its contents may be real or complex. This
%   method may be applied in the following ways:
%
%
%      Method 4: analytic problems in a large number of variables z and
%                large number of residuals F(z).
%      nls_gndl(f,dF,z0) where f(z) := 0.5*(F(z)'*F(z)) and dF is a
%      structure containing:
%
%         dF.JHF     - The function dF.JHF(zk) should return
%                      [dF(zk)/d(z^T)]'*F(zk), which is also equal to
%                      2*df(zk)/d(conj(z)) = 2*conj(df(zk)/d(z)) if z is
%                      complex, or equal to df(xk)/dx if it is real.
%         dF.JHJx    - The function dF.JHF(zk,x) should return the matrix-
%                      vector product ([dF(zk)/d(z^T)]'*[dF(zk)/d(z^T)])*x.
%
%   The structure output returns additional information:
%
%      output.alpha        - The plane search step lengths in every
%                            iteration, if a plane search is selected.
%      output.cgiterations - The number of CG/LSQR iterations to compute
%                            the Gauss-Newton step in every iteration.
%                            (large-scale methods only).
%      output.cgrelres     - The relative residual norm of the computed
%                            Gauss-Newton step (large-scale methods only).
%      output.delta        - The trust region radius at every step attempt.
%      output.fval         - The value of the objective function f in every
%                            iteration.
%      output.info         - The circumstances under which the procedure
%                            terminated:
%                               1: Objective function tolerance reached.
%                               2: Step size tolerance reached.
%                               3: Maximum number of iterations reached.
%      output.infops       - The circumstances under which the plane search
%                            terminated in every iteration.
%      output.iterations   - The number of iterations.
%      output.relfval      - The difference in objective function value
%                            between every two successive iterates,
%                            relativeto its initial value.
%      output.relstep      - The step size relative to the norm of the 
%                            current iterate in every iteration.
%      output.rho          - The trustworthiness at every step attempt.
%
%   nls_gndl(F,dF,z0,options) may be used to set the following options:
%
%      options.CGMaxIter = 15     - The maximum number of CG/LSQR
%                                   iterations for computing the
%                                   Gauss-Newton step (large-scale methods
%                                   only).
%      options.CGTol = 1e-6       - The tolerance for the CG/LSQR method to
%                                   compute the Gauss-Newton step
%                                   (large-scale methods only).
%      options.Delta = 'auto'     - The initial trust region radius. On
%                                   'auto', the radius is equal to the norm
%                                   of the first Gauss-Newton step.
%      options.Display = 1        - Displays the objective function value,
%                                   its difference with the previous
%                                   iterate relative to the first iterate
%                                   and the relative step size each
%                                   options.Display iterations. Set to 0 to
%                                   disable.
%      options.JHasFullRank       - If set to true, the Gauss-Newton step
%      = false                      is computed as a least squares
%                                   solution, if possible. Otherwise, it is
%                                   computed using a more expensive
%                                   pseudo-inverse.
%      options.MaxIter = 200      - The maximum number of iterations.
%      options.PlaneSearch        - The plane search used to minimize the
%      = false                      objective function in the plane spanned
%                                   by the steepest descent direction and
%                                   the Gauss-Newton step. Disables dogleg
%                                   trust region strategy. The method
%                                   should have the function signature
%                                   options.PlaneSearch(F,dF,z,p1,p2, ...
%                                   state,options.PlaneSearchOptions).
%      options.PlaneSearchOptions - The options structure passed to the
%                                   plane search search routine.
%      options.TolFun = 1e-12     - The tolerance for output.relfval. Note
%                                   that because the objective function is
%                                   a squared norm, TolFun can be as small
%                                   as eps^2.
%      options.TolX = 1e-6        - The tolerance for output.relstep.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, M. Van Barel, L. De Lathauwer, "Unconstrained
%       optimization of real functions in complex variables", SIAM J. Opt.,
%       Vol. 22, No. 3, 2012, pp. 879-898.

% Check the objective function f, derivative dF and first iterate z0.


% Evaluate the function value at z0.
dim = structure(z0);
z = z0;
z0 = serialize(z0);

fval = f(z);


% Numerical approximaton of complex derivatives.
function J = derivjac(zk)
    J = deriv(F,zk,Fval,type);
end

% In the case 'F+dFdzx+dFdconjzx', convert J*x and J'*x to the real domain.
function y = Jx(x,transp)
    x = x(1:end/2)+x(end/2+1:end)*1i;
    if strcmp(transp,'notransp')
        dFdzx = dF.dzx(z,x,transp);
        dFdconjzconjx = dF.dconjzx(z,conj(x),transp);
        y = [real(dFdzx)+real(dFdconjzconjx); ...
             imag(dFdzx)+imag(dFdconjzconjx)];
    else
        dFdzx = dF.dzx(z,x,transp);
        dFdconjzx = dF.dconjzx(z,x,transp);
        y = [real(dFdzx)+real(dFdconjzx); ...
             imag(dFdzx)-imag(dFdconjzx)];
    end
end


% In the case 'f+JHJx+JHF', compute JHJ*x.
function y = JHJx(x)
    y = dF.JHJx(z,x);
end

% Modify the preconditioner, if available.
function x = PC(b) % = M_blockJacobi
    x = dF.M(z,b);
end



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
while ~output.info

    % Compute the (in)exact Gauss-Newton step pgn.

    % Compute the Cauchy point pcp = -alpha*grad.
    grad = serialize(dF.JHF(z));
    gg = grad'*grad;
    gBg = real(grad'*dF.JHJx(z,grad));
    alpha = gg/gBg;
    if ~isfinite(alpha), alpha = 1; end;
    
    % Compute the Gauss-Newton step pgn.
    [pgn,~,output.cgrelres(end+1),output.cgiterations(end+1)] = ...
        mpcg(@JHJx,-grad,options.CGTol,options.CGMaxIter,dF.PC, ...
             [],-alpha*grad);

    rho = -inf;

    
    % Dogleg trust region.
    normpgn = norm(pgn);
    if isnan(output.delta(end)), output.delta(end) = max(1,normpgn); end
    while rho <= 0

        % Compute the dogleg step p.
        delta = output.delta(end);
        if normpgn <= delta
            p = pgn;
            dfval = -0.5*real(grad'*pgn);
        elseif alpha*sqrt(gg) >= delta
            p = (-delta/sqrt(gg))*grad;
            dfval = delta*(sqrt(gg)-0.5*delta/alpha);
        else
            bma = pgn+alpha*grad; bmabma = bma'*bma;
            a = -alpha*grad; aa = alpha^2*gg;
            c = real(a'*bma);
            if c <= 0
                beta = (-c+sqrt(c^2+bmabma*(delta^2-aa)))/bmabma;
            else
                beta = (delta^2-aa)/(c+sqrt(c^2+bmabma*(delta^2-aa)));
            end
            p = a+beta*bma;
            dfval = 0.5*alpha*(1-beta)^2*gg- ...
                    0.5*beta *(2-beta)*real(grad'*pgn);
        end

        % Compute the trustworthiness rho.
        if dfval > 0
            z = deserialize(z0+p,dim);
           
            fval = f(z);

            rho = (output.fval(end)-fval)/dfval;
            if isnan(rho), rho = -inf; end
            output.rho(end+1) = rho;
        end

        % Update trust region radius delta.
        if rho > 0.5
            output.delta(end+1) = max(delta,2*norm(p));
        else
            sigma = (1-0.25)/(1+exp(-14*(rho-0.25)))+0.25;
            if normpgn < sigma*delta && rho < 0
                e = ceil(log2(normpgn/delta)/log2(sigma));
                output.delta(end+1) = sigma^e*delta;
            else
                output.delta(end+1) = sigma*delta;
            end
        end
        
        % Check for convergence.
        relstep = norm(p)/norm(z0); if isnan(relstep), relstep = 0; end
        if rho <= 0 && relstep <= options.TolX
            output.rho(end+1) = rho;
            fval = output.fval(end);
            z = deserialize(z0,dim);
            break;
        end

    end

    % Save current state.
    if rho > 0
        z0 = serialize(z);
    end
    
    % Update the output structure.
    output.fval(end+1) = fval;
    output.iterations = output.iterations+1;
    output.relfval(end+1) = ...
        abs(diff(output.fval(end:-1:end-1)))/abs(output.fval(1));
    output.relstep(end+1) = relstep;
    if output.relfval(end) <= options.TolFun, output.info = 1; end
    if output.relstep(end) <= options.TolX, output.info = 2; end
    if output.iterations >= options.MaxIter, output.info = 3; end
end

function z = deserialize(z,dim)
    if iscell(dim)
        v = z; z = cell(size(dim));
        s = cellfun(@(s)prod(s(:)),dim(:)); o = [0; cumsum(s)];
        for i = 1:length(s), z{i} = reshape(v(o(i)+(1:s(i))),dim{i}); end
    elseif ~isempty(dim)
        z = reshape(z,dim);
    end
end

function z = serialize(z)
    if iscell(z)
        s = cellfun(@numel,z(:)); o = [0; cumsum(s)];
        c = z; z = zeros(o(end),1);
        for i = 1:length(s), ci = c{i}; z(o(i)+(1:s(i))) = ci(:); end
    else
        z = z(:);
    end
end

function dim = structure(z)
    if iscell(z)
        dim = cellfun(@size,z,'UniformOutput',false);
    else
        dim = size(z);
        if numel(z) == dim(1), dim = []; end
    end
end
'''

