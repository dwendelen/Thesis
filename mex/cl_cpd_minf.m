function [U,output] = cpd_minf(T,U0,options)
%CPD_MINF CPD by unconstrained nonlinear optimization.
%   [U,output] = cpd_minf(T,U0) computes the factor matrices U{1}, ...,
%   U{N} belonging to a canonical polyadic decomposition of the N-th order
%   tensor T by minimizing 0.5*frob(T-cpdgen(U))^2. The algorithm is
%   initialized with the factor matrices U0{n}. The structure output
%   returns additional information:
%
%      output.Name  - The name of the selected algorithm.
%      output.<...> - The output of the selected algorithm.
%
%   cpd_minf(T,U0,options) may be used to set the following options:
%
%      options.Algorithm =     - The desired optimization method.
%      [{@minf_lbfgsdl}|...
%       @minf_lbfgs|@minf_ncg]
%      options.LineSearch =    - A function handle to the desired line
%      [{'auto'},@ls_mt|...      search algorithm. If the line search
%       @cpd_aels|@cpd_els|...   method has the prefix 'cpd_', it is
%       @cpd_lsb]                modified to be compatible with the
%                                optimization algorithm. Only applicable to
%                                algorithms with line search globalization.
%      options.PlaneSearch =   - A function handle to the desired CPD plane
%      [{false},@cpd_eps]        search algorithm. Only applicable to
%                                algorithms with trust region
%                                globalization.
%      options.<...>           - Parameters passed to the selected method,
%                                e.g., options.TolFun, options.TolX and
%                                options.LineSearchOptions. See also help
%                                [options.Algorithm].
%
%   See also cpd_nls.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, M. Van Barel, L. De Lathauwer, "Optimization-based
%       algorithms for tensor decompositions: canonical polyadic
%       decomposition, decomposition in rank-(Lr,Lr,1) terms and a new
%       generalization," SIAM J. Opt., 2013.
%   [2] L. Sorber, M. Van Barel, L. De Lathauwer, "Unconstrained
%       optimization of real functions in complex variables," SIAM J. Opt.,
%       Vol. 22, No. 3, 2012, pp. 879-898.

% Check the tensor T.
N = ndims(T);
if N < 3, error('cpd_minf:T','ndims(T) should be >= 3.'); end

% Check the initial factor matrices U0.
U = U0(:).';
R = size(U{1},2);
size_tens = size(T);



if any(cellfun('size',U,2) ~= R)
    error('cpd_minf:U0','size(U0{n},2) should be the same for all n.');
end
if any(cellfun('size',U,1) ~= size_tens)
    error('cpd_minf:U0','size(T,n) should equal size(U0{n},1).');
end

% Check the options structure.
isfunc = @(f)isa(f,'function_handle');
xsfunc = @(f)isfunc(f)&&exist(func2str(f),'file');
if nargin < 3, options = struct; end
if ~isfield(options,'Algorithm')
    funcs = {@minf_lbfgsdl,@minf_lbfgs,@minf_ncg};
    options.Algorithm = funcs{find(cellfun(xsfunc,funcs),1)};
end
if ~isfield(options,'Display'), options.Display = 0; end
if ~isfield(options,'TolFun'), options.TolFun = 1e-12; end

% Adapt line/plane search if it is a CPD line/plane search.
if isfield(options,'LineSearch') && ...
   ~isempty(strfind(func2str(options.LineSearch),'cpd_'))
    linesearch = options.LineSearch;
    options.LineSearch = @ls;
end
if isfield(options,'PlaneSearch') && ...
   ~isempty(strfind(func2str(options.PlaneSearch),'cpd_'))
    planesearch = options.PlaneSearch;
    options.PlaneSearch = @ps;
end

% Cache some intermediate variables.
M = arrayfun(@(n)tens2mat(T,n),1:N,'UniformOutput',false);
UHU = zeros(R,R,N);
T2 = T(:)'*T(:);

cl_cpd_gateway('setTAndU', T, U0);

% Call the optimization method.
[U,output] = options.Algorithm(@f,@g,U,options);
output.Name = func2str(options.Algorithm);

function u(U)
    cl_cpd_gateway('setU', U);
end
function s = r()
    s = cl_cpd_gateway('run');
end
function fval = f(U)
    u(U);
    s = r();
    fval = 0.5 * sum(s);
end

function grad = g(U)
    grad = cell(1,N);
    for n = 1:N, UHU(:,:,n) = U{n}'*U{n}; end
    for n = 1:N
        G1 = U{n}*conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        G2 = M{n}*conj(kr(U([N:-1:n+1 n-1:-1:1])));
        grad{n} = G1-G2;
    end
end

function [alpha,output] = ls(~,~,z,p,state,options)
    state.UHU = UHU;
    state.T2 = T2;
    [alpha,output] = linesearch(T,z,p,state,options);
end

function [alpha,output] = ps(~,~,z,p,q,state,options)
    state.UHU = UHU;
    state.T2 = T2;
    [alpha,output] = planesearch(T,z,p,q,state,options);
end

end
