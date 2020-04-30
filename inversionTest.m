clear
clc
close all

N = 100;
x = rand(N,1);
X = diag(x);

tic
Xinv = eye(N)/X;
toc


tic
Xinv2 = zeros(N);
for ii = 1:N
    Xinv2(ii,ii) = 1/x(ii);
end
toc