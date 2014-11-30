
function cl_cpd_setU(U)
    if nargin < 1
        error('please provide U')
    end
    
    if ~iscell(U)
        error('U must be a cell')
    end
    
    if 3 ~= length(U)
        error('Currently only 3-dimensional tensors are supported')
    end
    
    for i = length(U)
        if ~isnumeric(U{i})
            error('The cells of U must all be numerical arrays')
        end
    end
    
    cl_cpd_gateway('setU', U)
end

