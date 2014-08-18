compile

N = 3;
R = 16;
T = rand(208,160,64);
U = cpd_rnd(size(T), R);

M = arrayfun(@(n)tens2mat(T,n),1:N,'UniformOutput',false);
D = M{1}-U{1}*kr(U(end:-1:2)).';
e = sum(D(:)'*D(:));

UHU = zeros(R,R,N);
grad = cell(1,N);
for n = 1:N, UHU(:,:,n) = U{n}'*U{n}; end
for n = 1:N
    G1 = U{n}*conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
    G2 = M{n}*conj(kr(U([N:-1:n+1 n-1:-1:1])));
    grad{n} = G1-G2;
end

x = cl_cpd_gateway('test', T, U, e, 600, grad, 0)