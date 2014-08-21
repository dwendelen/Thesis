T = rand(128, 128, 128);
U1 = cpd_rnd(size(T), 200);
U2 = cpd_rnd(size(T), 200);

cl_cpd_gateway('init', true);

N = 3;

tic
cpd_minf(T, U1)
toc

tic
cl_cpd_minf(T, U1)
toc



