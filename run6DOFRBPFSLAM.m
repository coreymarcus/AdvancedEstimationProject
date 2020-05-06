% Corey Marcus
% Advanced Estimation
% This script runs a rbpf on a full 6DOF SLAM model

clear
close all
clc

%% Approximate Covariance for Angular Process Noise
N = 100000; %number of samples
Peuler = .0001*eye(3);
muEuler = [0 0 0]';

quatMat = zeros(N,4);
for ii = 1:N
    E = mvnrnd(muEuler,Peuler);
    quatMat(ii,:) = angle2quat(E(1),E(2),E(3),'ZYX');
end

muQuat = mean(quatMat,1);
covQuat = cov(quatMat);
% save('quatParams.mat','muQuat','covQuat','Peuler');
% load('quatParams.mat');

%% System
dt = .25;
t = 0:dt:20*dt;
L = length(t);
dim = 3; %dimension of the model
Nstate = 13; %dimension of nonlinear state (pos, vel, quat inertial to body, and rate wrt inertial expressed in body)
Nmap = 5; %number of map objects
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
sys.Pnu_n = blkdiag(.05*eye(3),.01*eye(3),covQuat,.01*eye(3));
sys.Peta = 0.1*eye(dim*Nmap);
sys.N_n = Nstate;
sys.N_l = dim*Nmap;
sys.Peuler = Peuler;

%% Filter Parameters
Npart = 3000;
Params.Npart = Npart;
Params.estimateAngles = true;


%% Truth Initialization
mapBounds = [-10 10];
muMap = mapBounds(1) + (mapBounds(2) - mapBounds(1))*rand(dim*Nmap,1);
Pmap0 = 0.1*eye(dim*Nmap);
muTheta0 = zeros(3,1);
Ptheta0 = Peuler;
mapTruth = mvnrnd(muMap, Pmap0)';
mapTruthX = mapTruth(1:3:(dim*Nmap - 2));
mapTruthY = mapTruth(2:3:(dim*Nmap - 1));
mapTruthZ = mapTruth(3:3:(dim*Nmap - 0));
muPose = zeros(Nstate,1);
muPose(4:6) = [1 1 1]';
muPose(1:3) = [-10 -10 -10];
muPose(11:13) = [.1 .1 .1];
Ppose0 = sys.Pnu_n;
thetaTruth0 = mvnrnd(muTheta0,Ptheta0)';
%using a scalar first quaternion representation
poseTruth0(7:10) = angle2quat(thetaTruth0(1),thetaTruth0(2),thetaTruth0(3))';
poseTruth0([1:6 11:13]) = mvnrnd(muPose([1:6 11:13]), Ppose0([1:6 11:13], [1:6 11:13]))';

%% Filter Initialization

%create the particle linear estimate and weight
p0.w = 1/Npart;
p0.xHat_l = muMap;
p0.P_l = Pmap0;

%create all the particles
xHat = cell(Npart,1);
for ii = 1:Npart
    p0.xHat_n = zeros(Nstate,1);
    p0.xHat_n([1:6 11:13]) = mvnrnd(muPose([1:6 11:13]), Ppose0([1:6 11:13], [1:6 11:13]))';
    
    %initialize attitude
    thetaParticle = mvnrnd(muTheta0, Ptheta0)';
    p0.xHat_n(7:10) = angle2quat(thetaParticle(1),thetaParticle(2),thetaParticle(3))';
    
    %assign
    xHat{ii} = p0;
end

%% Run

% storing the estimate
xHatMat = cell(Npart,L);
xHatMat(:,1) = xHat;
xHatMat_l = zeros(sys.N_l,L);
xHatMat_n = zeros(sys.N_n,L);

%storing the truth
poseTruth = zeros(sys.N_n, L);
poseTruth(:,1) = poseTruth0;

%weights
wMat = zeros(Params.Npart, L);
wMat(:,1) = p0.w*ones(Params.Npart,1);

%effective number of particles
Neff = zeros(1,L);
Neff(1) = 1/sum(wMat(:,1).^2);

%vectors for alignment of body
vMatTruth = zeros(3,L);
vMatEst = zeros(3,L);
v = [1 0 0]';
vMatTruth(:,1) = quatrotate(poseTruth(7:10,1)',v')';


for ii = 2:L
    
    %Propagate Dynamics
    poseTruth(:,ii) = sys.f_n(poseTruth(:,ii-1));
    muQuat = poseTruth(7:10,ii);
    poseTruth(:,ii) = poseTruth(:,ii) + mvnrnd(zeros(sys.N_n,1),sys.Pnu_n)';
    
     %draw random euler angles and create new quaternion
    randEuler = mvnrnd([0 0 0]', sys.Peuler);
    quatNoise = angle2quat(randEuler(1),randEuler(2),randEuler(3),'ZYX');
    poseTruth(7:10,ii) = quatmultiply(muQuat', quatNoise)';
    vMatTruth(:,ii) = quatrotate(poseTruth(7:10,ii)',v')';
    
    %generate a measurement
    eta = mvnrnd(zeros(sys.N_l,1),sys.Peta)';
    y = sys.h(poseTruth(:,ii)) + sys.C(poseTruth(:,ii))*mapTruth + sys.D(poseTruth(:,ii))*eta;
    
    %run RB Particle Filter
    [xHat, xHat_l, xHat_n] = rbpfSLAM(sys, y, xHat, Params);
    
    %store values
    xHatMat(:,ii) = xHat;
    xHatMat_l(:,ii) = xHat_l;
    xHatMat_n(:,ii) = xHat_n;
    for kk = 1:Params.Npart
        wMat(kk,ii) = xHat{kk}.w;
    end
    vMatEst(:,ii) = quatrotate(xHatMat_n(7:10,ii)',v')';
    
    %calculate effective number of particles
    Neff(ii) = 1/sum(wMat(:,ii).^2);
    
    disp(Neff(ii));
    disp(ii/L);
    
end

%get first estimate
for ii = 1:Npart
    xHatMat_l(:,1) = xHatMat_l(:,1) + xHatMat{ii,1}.w * xHatMat{ii,1}.xHat_l;
    xHatMat_n(:,1) = xHatMat_n(:,1) + xHatMat{ii,1}.w * xHatMat{ii,1}.xHat_n;
end
xHatMat_n(7:10,1) = xHatMat_n(7:10,1)/norm(xHatMat_n(7:10,1));
vMatEst(:,1) = quatrotate(xHatMat_n(7:10,1)',v')';

%extract map estimate
mapHatXMat = xHatMat_l(1:dim:(dim*Nmap - 2),:);
mapHatYMat = xHatMat_l(2:dim:(dim*Nmap - 1),:);
mapHatZMat = xHatMat_l(3:dim:(dim*Nmap - 0),:);

%mapping error
eMapX = mapHatXMat - mapTruthX;
eMapY = mapHatYMat - mapTruthY;
eMapZ = mapHatZMat - mapTruthZ;
eMap = sqrt(mean(eMapX.^2 + eMapY.^2 + eMapZ.^2,1));

%Calculate the error for each particle at each instant in time
eNonLin = zeros(Nstate,Npart,L);
for ii = 1:L
    targ = xHatMat(:,ii);
    for jj = 1:Npart
%         eNonLin(:,jj,ii) = poseTruth(:,ii) - targ{jj}.xHat_n;
                eNonLin(:,jj,ii) = targ{jj}.xHat_n;
    end
end

%% Align truth and estimate
estMapVect = xHatMat_l(:,end);
% for ii = 1:Params.Npart
%     estMapVect(3*ii-2:3*ii) = xHatMat_l(1:3,end);
% end
% 
% for ii = 1:Params.Npart
%     
% end

%cost
cost = @(x)alignCost(x,estMapVect,mapTruth);

options = optimoptions('lsqnonlin','MaxFunctionEvaluations',100000, ...
    'MaxIterations',100000,'Algorithm','levenberg-marquardt',...
    'FunctionTolerance',1E-8,'Display','iter');

%calculate alignment
xHat = lsqnonlin(cost,zeros(6,1),[],[],options);
That = xHat(1:3);
Rhat = angle2dcm(xHat(4),xHat(5),xHat(6),'ZYX');

estPlot = zeros(3,L);
for ii = 1:L
    estPlot(:,ii) = Rhat*(xHatMat_n(1:3,ii) + That);
end

estMap = zeros(3, Nmap);
for ii = 1:Nmap
    estMap(:,ii) = Rhat*(xHatMat_l(3*ii-2:3*ii,end) + That);
end

% figure
% plot3(poseTruth(1,:),poseTruth(2,:),poseTruth(3,:))
% hold on
% plot3(estPlot(1,:),estPlot(2,:),estPlot(3,:))
% legend('True','Estimate')

figure
scatter3(mapTruthX, mapTruthY, mapTruthZ)
hold on
scatter3(estMap(1,:),estMap(2,:),estMap(3,:))
legend('True','Estimate')


%% Plotting

% figure
% subplot(4,1,1)
% plot(t,xHatMat_n(1,:))
% hold on
% plot(t,poseTruth(1,:))
% ylabel('x1')
% legend('Est Traj.','True Traj')
% subplot(4,1,2)
% plot(t,xHatMat_n(2,:))
% hold on
% plot(t,poseTruth(2,:))
% ylabel('x2')
% subplot(4,1,3)
% plot(t,xHatMat_n(3,:))
% hold on
% plot(t,poseTruth(3,:))
% ylabel('x3')
% legend('Est Traj.','True Traj')
% subplot(4,1,4)
% plot(t,xHatMat_n(4,:))
% hold on
% plot(t,poseTruth(4,:))
% ylabel('x4')
% xlabel('Time')
% legend('Est Traj.','True Traj')

figure
plot3(estPlot(1,:), estPlot(2,:), estPlot(3,:))
hold on
plot3(poseTruth(1,:), poseTruth(2,:), poseTruth(3,:))
quiver3(estPlot(1,:), estPlot(2,:), estPlot(3,:),...
    vMatEst(1,:), vMatEst(2,:), vMatEst(3,:))
quiver3(poseTruth(1,:), poseTruth(2,:), poseTruth(3,:),...
    vMatTruth(1,:), vMatTruth(2,:), vMatTruth(3,:))
legend('Est Traj.','True Traj','Est Pointing', 'True Pointing')
xlabel('x')
ylabel('y')
zlabel('z')

figure
plot(t, estPlot(1,:), t, poseTruth(1,:))
legend('Est Traj.','True Traj','Est Pointing', 'True Pointing')
xlabel('Time')
ylabel('x')

figure
plot(t, estPlot(2,:), t, poseTruth(2,:))
legend('Est Traj.','True Traj','Est Pointing', 'True Pointing')
xlabel('Time')
ylabel('y')

figure
plot(t, estPlot(3,:), t, poseTruth(3,:))
legend('Est Traj.','True Traj','Est Pointing', 'True Pointing')
xlabel('Time')
ylabel('z')

figure
plot(t,wMat)
xlabel('Time')
ylabel('Particle Weights')

figure
plot(t, Neff)
xlabel('Time')
ylabel('Effective Number of Particles')

figure
scatter(mapHatXMat(:,end),mapHatYMat(:,end))
hold on
scatter(mapTruthX,mapTruthY)
xlabel('x')
ylabel('y')
title('Mapping Performance')
legend('Estimated Map','True Map')

figure
scatter(mapHatXMat(:,end) - mapTruthX, mapHatYMat(:,end) - mapTruthY)
title('Mapping Error')
xlabel('x')
ylabel('y')
lims = abs(axis);
axis([-max(lims(1:2)) max(lims(1:2)) -max(lims(3:4)) max(lims(3:4))]);

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
