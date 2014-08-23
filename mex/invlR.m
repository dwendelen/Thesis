%compile
%[Data_invlR] = cl_cpd_gateway('InvlR');
%save Data_invlR;

load Data_invlR;

g = Data_invlR(1);

X = g.lines(1).x
G64 = (2 * X * (64^2 + 64^3) + 2*(64^3))./ g.lines(1).y;
G128 = (2 * X * (128^2 + 128^3) + 2*(128^3))./ g.lines(2).y;
G320 = (2 * X * (320^2 + 320^3) + 2*(320^3))./ g.lines(3).y;

hold on;
    semilogy(X, G64, 'b')
    semilogy(X, G128, 'r')
    semilogy(X, G320, 'g')
hold off;
legend('I = 64', 'I = 128', 'I = 320')