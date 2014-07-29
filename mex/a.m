compile
T = zeros(16, 16, 16);
U1 = cell(1, 3)
U1{1} = zeros(16, 5)
U1{2} = zeros(16, 5)
U1{3} = zeros(16, 5)

cl_cpd_gateway('init', true)
cl_cpd_gateway('setTAndU', T, U1)
x = cl_cpd_gateway('run')

U2 = cell(1,3);
U2{1} = zeros(16, 5);
U2{2} = zeros(16, 5);
U2{3} = zeros(16, 5);

U2{1}(1,1) = 1;
U2{2}(1,1) = 2;
U2{3}(1,1) = 3;

cl_cpd_gateway('setU', U2)
x = cl_cpd_gateway('run')