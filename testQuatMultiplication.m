clear
close all
clc


q = angle2quat(.1, -.1, .3);

Peuler = .005*eye(3);

N = 100;

eQuat = zeros(N,4);

for ii = 1:N
    eAngle = mvnrnd([0 0 0]',Peuler)';
    eQuat(ii,:) = angle2quat(eAngle(1), eAngle(2), eAngle(3));
    
end

prod1 = quatmultiply(q,eQuat);
prod2 = quatmultiply(eQuat,q);

v = [1 0 0];

V1 = quatrotate(prod1,v);
V2 = quatrotate(prod2,v);

figure
scatter3(V1(:,1),V1(:,2),V1(:,3))
hold on
scatter3(V2(:,1),V2(:,2),V2(:,3))
axis([-2 2 -2 2 -2 2])