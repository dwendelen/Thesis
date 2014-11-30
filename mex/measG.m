s = size(Data_measureG_data);
T = zeros(s(2,3));
T(:,:) = Data_measureG_data(1,:,:);

II = 16:16:320;
RR = 16:16:1024;

[R, I] = meshgrid(RR, II);

G = 2*R.*(I.*I + I.*I.*I);

contourf(R,I,G)