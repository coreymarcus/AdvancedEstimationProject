% This script used to approximate the covariance of the quaternion given
% the covariance of the euler angles

clear
close all
clc

N = 100000; %number of samples
Peuler = 0.01*eye(3);
muEuler = [0 0 0]';

quatMat = zeros(N,4);
for ii = 1:N
    E = mvnrnd(muEuler,Peuler);
    quatMat(ii,:) = angle2quat(E(1),E(2),E(3),'ZYX');
end

muQuat = mean(quatMat,1);
covQuat = cov(quatMat);

save('quatParams.mat','muQuat','covQuat','Peuler');
