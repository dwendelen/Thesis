function cl_cpd_init(profile)
    if nargin < 1
        profile = false;
    end

    if ~islogical(profile)
        error('profile must be a boolean')
    end

    cl_cpd_gateway('init', profile)
end

