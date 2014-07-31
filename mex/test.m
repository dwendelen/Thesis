T = rand(32,32,32);
U1 = cpd_rnd(size(T), 10);
U2 = cpd_rnd(size(T), 10);

cl_cpd_gateway('init', true);
tic
cl_cpd_gateway('setTAndU', T, U1);
s = cl_cpd_gateway('run');
e1 = sum(s)
toc
t = cl_cpd_gateway('time')/1000000
tic
cl_cpd_gateway('setU', U2);
s = cl_cpd_gateway('run');
e1 = sum(s)
toc
t = cl_cpd_gateway('time')/1000000

N = 3;
M = arrayfun(@(n)tens2mat(T,n),1:N,'UniformOutput',false);
tic
D = M{1}-U1{1}*kr(U1(end:-1:2)).';
sum(D(:)'*D(:))
toc