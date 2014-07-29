function do(R, I)

    I = ceil(I/16)*16;

    r = R
    i = I
    
    T = rand(I, I, I);
    U = rand(I, R);
    
    a = cell(1,3);
    a{3} = U;
    a{1} = U;
    a{2} = U;
    
    cl_cpd_gateway('setTAndU', T, a)

    run('Version UnRemapped')
    %%run(i, (t0, t1, t2), 'Version Isolated')
    %run(r, (t0, t1, t2), 'Version ReMapped')
    %b.release()
    %run(r8, (t0, t1, t2), 'Version 8x8x8')
    %b8.release()
    %run(r4, (t0, t1, t2), 'Version 4x4x4')
    %b4.release()

end

