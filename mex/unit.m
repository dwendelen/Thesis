compile

N = 3;
T = rand(208,160,64);
U = cpd_rnd(size(T), 16);

M = arrayfun(@(n)tens2mat(T,n),1:N,'UniformOutput',false);
D = M{1}-U{1}*kr(U(end:-1:2)).';
e = sum(D(:)'*D(:));

x = cl_cpd_gateway('test', T, U, e, 600)