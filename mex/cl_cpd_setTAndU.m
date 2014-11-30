function cl_cpd_setTAndU(T, U)
    if nargin < 2
        error('please provide T and U')
    end
    
    if ~isnumeric(T)
        error('T must be an array with numerical elements')
    end
    
    if ~iscell(U)
        error('U must be a cell')
    end
    
    nbDims = ndims(T)
    dims = size(T)
    
    if nbDims ~= 3
        error('Currently only 3-dimensional tensors are supported')
    end
    
    if nbDims ~= length(U)
        error('Dimensions T and U do not match')
    end
    
    for i = length(U)
        if ~isnumeric(U{i})
            error('The cells of U must all be numerical arrays')
        end
        sizeU = size(U{i})
        if dims(i) ~= sizeU(1)
            error('Dimensions T and U do not match')
        end
    end
    
    cl_cpd_gateway('setTAndU', T, U)
end

