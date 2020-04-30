% A quick script to test my 6DOF model
clear
close all
clc

%% System
dt = .1;
t = 0:dt:10;
L = length(t);
dim = 3; %dimension of the model
Nstate = 13; %dimension of nonlinear state (pos, vel, quat inertial to body, and rate wrt inertial expressed in body)
Nmap = 2; %number of map objects
Acam = eye(Nstate);
Acam(1,3) = dt;
Acam(2,4) = dt;
sys.f_n = @(x_n) f_n(x_n,dt); %nonlinear 6DOF dynamics
sys.B_n = @(x_n) eye(Nstate);
sys.A_l = @(x_n) eye(dim*Nmap);
sys.B_l = @(x_n) eye(dim*Nmap);
sys.h = @(x_n) h(x_n, Nmap);
sys.C = @(x_n) C(x_n, Nmap);
sys.D = @(x_n) eye(dim*Nmap);
sys.Pnu_n = .01*eye(Nstate);
sys.Peta = .01*eye(dim*Nmap); %map assumed static
sys.N_n = Nstate;
sys.N_l = dim*Nmap;


%% Truth Initialization
mapBounds = [-10 10];
muMap = mapBounds(1) + (mapBounds(2) - mapBounds(1))*rand(dim*Nmap,1);
Pmap0 = eye(dim*Nmap);
muTheta0 = zeros(3,1);
Ptheta0 = 0.01*eye(3);
mapTruth = mvnrnd(muMap, Pmap0)';
mapTruthX = mapTruth(1:3:(dim*Nmap - 2));
mapTruthY = mapTruth(2:3:(dim*Nmap - 1));
mapTruthZ = mapTruth(3:3:(dim*Nmap - 0));
muPose = zeros(Nstate,1);
muPose(3:4) = [1 1]';
Ppose0 = .1*eye(Nstate);
poseTruth0 = zeros(Nstate,1);
thetaTruth0 = mvnrnd(muTheta0,Ptheta0)';
%using a scalar first quaternion representation
poseTruth0(7:10) = angle2quat(thetaTruth0(1),thetaTruth0(2),thetaTruth0(3))';
poseTruth0([1:6 11:13]) = mvnrnd(muPose([1:6 11:13]),Ppose0([1:6 11:13], [1:6 11:13]))';

%% Propagation

%initialization
poseTruth = zeros(sys.N_n, L);
poseTruth(:,1) = poseTruth0;
v0 = [1 0 0]';
vHist = zeros(3,L);
vHist(:,1) = quatrotate(quatconj(poseTruth(7:10,1)'),v0')';
yHist = zeros(3*Nmap, L);
yHistInert = zeros(3*Nmap, L);

%first measurement
y = sys.h(poseTruth(:,1)) + sys.C(poseTruth(:,1))*mapTruth;
yHist(:,1) = y;
yHistInert(1:3,1) = quatrotate(quatconj(poseTruth(7:10,1)'),y(1:3)')';
yHistInert(4:6,1) = quatrotate(quatconj(poseTruth(7:10,1)'),y(4:6)')';

% loop
for ii = 2:L
    
    %Propagate Dynamics
    poseTruth(:,ii) = sys.f_n(poseTruth(:,ii-1));
    
    %quaternion
    qBody2Inert = quatconj(poseTruth(7:10,ii)');
    
    %create vector of body Directions
    vHist(:,ii) = quatrotate(qBody2Inert,v0')';
    
    %measurement
    y = sys.h(poseTruth(:,ii)) + sys.C(poseTruth(:,ii))*mapTruth;
    yHist(:,ii) = y;
    
    %convert measurement to inertial
    yHistInert(1:3,ii) = quatrotate(qBody2Inert,y(1:3)')';
    yHistInert(4:6,ii) = quatrotate(qBody2Inert,y(4:6)')';
    
    
end

%% Plotting
figure
plot3(poseTruth(1,:), poseTruth(2,:), poseTruth(3,:))
hold on
quiver3(poseTruth(1,:), poseTruth(2,:), poseTruth(3,:),...
    vHist(1,:),vHist(2,:),vHist(3,:));
xlabel('x')
ylabel('y')
zlabel('z')
title('Position')
scatter3(mapTruthX,mapTruthY,mapTruthZ);
quiver3(poseTruth(1,:), poseTruth(2,:), poseTruth(3,:),...
    yHistInert(1,:),yHistInert(2,:),yHistInert(3,:),0);
quiver3(poseTruth(1,:), poseTruth(2,:), poseTruth(3,:),...
    yHistInert(4,:),yHistInert(5,:),yHistInert(6,:),0);

figure
subplot(3,1,1)
plot(t,poseTruth(4,:))
title('Velocity')
ylabel('v_x')
subplot(3,1,2)
plot(t,poseTruth(5,:))
ylabel('v_y')
subplot(3,1,3)
plot(t,poseTruth(6,:))
ylabel('v_z')
xlabel('time')

figure
subplot(3,1,1)
title('Euler Angles')
plot(t,unwrap(poseTruth(7,:)))
ylabel('yaw')
subplot(3,1,2)
plot(t,unwrap(poseTruth(8,:)))
ylabel('pitch')
subplot(3,1,3)
plot(t,unwrap(poseTruth(9,:)))
ylabel('roll')
xlabel('time')

figure
subplot(3,1,1)
title('Rates')
plot(t,poseTruth(10,:))
ylabel('yaw')
subplot(3,1,2)
plot(t,poseTruth(11,:))
ylabel('pitch')
subplot(3,1,3)
plot(t,poseTruth(12,:))
ylabel('roll')
xlabel('time')


%% Functions

%propagation function
function xOut = f_n(x, dt)

%propagate
[~, propOut] = ode45(@(t,xDummy) f(t,xDummy), [0 dt], x);
xOut = propOut(end,:)';

%renormalize quaternion
xOut(7:10) = xOut(7:10)/norm(xOut(7:10));

end


%dynamics function for ode45
function dxdt = f(~,x)

%locals
v = x(4:6);
q = x(7:10);
w = x(11:13);

%output
dxdt = zeros(13,1);
dxdt(1:3) = v;

%change in quat
dxdt(7:10) = quatmultiply(q',[0,0.5*w'])';

end

%measurement function
function y = h(x, Nmap)

%locals
r = x(1:3);
q = x(7:10);

%rotation matrix from inetial to body
R = quat2dcm(q');

%output
y = repmat(-R*r,Nmap,1);

end

%mapping between linear states and measurement
function Cout = C(x_n, Nmap)

%locals
q = x_n(7:10);

%rotation matrix from inetial to body
R = quat2dcm(q');

%output
Rcell = repmat({R},1,Nmap);
Cout = blkdiag(Rcell{:});

end