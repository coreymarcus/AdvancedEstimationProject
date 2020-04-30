function J = alignCost(x, Estimate, Truth)
%alignCost is used to find the rotation and translation between the
%estimate and the truth. x is [T E] where e are euler angles from estimate
%to truth in a 'ZYX' order)

%locals
N = length(Estimate)/3; %number of points
T = x(1:3);
E = x(4:6);
R = angle2dcm(E(1),E(2),E(3),'ZYX');

%rotate and translate each point in estimate into the truth
TruthHat = zeros(3*N,1);
for ii = 1:N
    p = Estimate(3*ii - 2: 3*ii);
    pHat = R*(p + T);
    TruthHat(3*ii - 2: 3*ii) = pHat;    
end

%calculate cost
J = norm(TruthHat - Truth);

end

