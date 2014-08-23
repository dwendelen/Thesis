a = zeros(5, 4, 40);
a(:) = Data_measureF(:);
I = 16:16:320;

i = 2

R = [8, 16, 400,4000]

T = a(:,1,1:2:39);

A1 = T(1,:);
B1 = T(2,:);
C1 = T(3,:);
D1 = T(4,:);
E1 = T(5,:);

T = a(:,2,1:2:39);

A2 = T(1,:);
B2 = T(2,:);
C2 = T(3,:);
D2 = T(4,:);
E2 = T(5,:);

T = a(:,3,1:2:39);

A3 = T(1,:);
B3 = T(2,:);
C3 = T(3,:);
D3 = T(4,:);
E3 = T(5,:);

T = a(:,4,1:2:39);

A4 = T(1,:);
B4 = T(2,:);
C4 = T(3,:);
D4 = T(4,:);
E4 = T(5,:);



F1 = 2*R(1)*(I.*I + I.*I.*I) + 2*I.*I.*I;
F2 = 2*R(2)*(I.*I + I.*I.*I) + 2*I.*I.*I;
F3 = 2*R(3)*(I.*I + I.*I.*I) + 2*I.*I.*I;
F4 = 2*R(4)*(I.*I + I.*I.*I) + 2*I.*I.*I;


AA1 = F1./A1;
BB1 = F1./B1;
CC1 = F1./C1;
DD1 = F1./D1;
EE1 = F1./E1;

AA2 = F2./A2;
BB2 = F2./B2;
CC2 = F2./C2;
DD2 = F2./D2;
EE2 = F2./E2;

AA3 = F3./A3;
BB3 = F3./B3;
CC3 = F3./C3;
DD3 = F3./D3;
EE3 = F3./E3;

AA4 = F4./A4;
BB4 = F4./B4;
CC4 = F4./C4;
DD4 = F4./D4;
EE4 = F4./E4;

hold on
plot(I, AA1, I, BB1, I, CC1, I, DD1, I, EE1)
plot(I, AA2, I, BB2, I, CC2, I, DD2, I, EE2)
plot(I, AA3, I, BB3, I, CC3, I, DD3, I, EE3)
plot(I, AA4, I, BB4, I, CC4, I, DD4, I, EE4)
legend('8, 16', '8,R', '8,I','16,16', '16,R', '16,I','400,16', '400,R', '400,I','4000,16', '4000,R', '4000,I')
hold off