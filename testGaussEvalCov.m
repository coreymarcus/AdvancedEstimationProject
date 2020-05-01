clear
close all
clc

% In theory, the covariance of the gaussian should be immaterial when
% considering the relative liklihood of multiple samples. I will try to
% test that theory here


mu = [1;
    0];
y = [.3 .5 2;
    .2 0 -1];
P = eye(2);

w1 = zeros(1,3);
w2 = zeros(1,3);
a = 100;
for ii = 1:3
    w1(ii) = gaussEval(y(:,ii),mu,P);
    w2(ii) = gaussEval(y(:,ii),mu,a*P);
end

w1 = w1/sum(w1);
w2 = w2/sum(w2);
w2 = w2.^a;
w2 = w2/sum(w2);


disp(w1)
disp(w2)
