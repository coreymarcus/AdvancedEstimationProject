% Corey Marcus
% Advanced Estimation
% This script runs a rbpf on a 3DOF SLAM model

clear
close all
clc

%% System
dt = .1;
t = 0:dt:5;
L = length(t);
dim = 3; %dimension of the model
Nstate = 6; %dimension of nonlinear state (pos, vel)
Nmap = 25; %number of map objects
sys.f_n = @(x_n) f_n(x_n,dt); %nonlinear dynamics
sys.B_n = @(x_n) eye(Nstate);
sys.A_l = @(x_n) eye(dim*Nmap);
sys.B_l = @(x_n) eye(dim*Nmap);
sys.h = @(x_n) h(x_n, Nmap);
sys.C = @(x_n) eye(dim*Nmap);
sys.D = @(x_n) eye(dim*Nmap);
sys.Pnu_n = 1*eye(Nstate);
sys.Peta = 0.1*eye(dim*Nmap);
sys.N_n = Nstate;
sys.N_l = dim*Nmap;

%% Filter Parameters
Npart = 1000;
Params.Npart = Npart;
Params.estimateAngles = false;


%% Truth Initialization
mapBounds = [-10 10];
muMap = mapBounds(1) + (mapBounds(2) - mapBounds(1))*rand(dim*Nmap,1);
Pmap0 = 0*eye(dim*Nmap);
% mapTruth = mvnrnd(muMap, Pmap0)';
mapTruth = muMap;
mapTruthX = mapTruth(1:3:(dim*Nmap - 2));
mapTruthY = mapTruth(2:3:(dim*Nmap - 1));
mapTruthZ = mapTruth(3:3:(dim*Nmap - 0));
muPose = zeros(Nstate,1);
muPose(4:5) = [1 1]';
Ppose0 = 1.0*eye(Nstate);
poseTruth0 = mvnrnd(muPose, Ppose0)';

%% Filter Initialization

%create the particle linear estimate and weight
p0.w = 1/Npart;
p0.xHat_l = muMap;
p0.P_l = Pmap0;

%create all the particles
xHat = cell(Npart,1);
for ii = 1:Npart
    p0.xHat_n = zeros(Nstate,1);
    p0.xHat_n = mvnrnd(muPose, Ppose0)';
        
    %assign
    xHat{ii} = p0;
end

%% Run
xHatMat = cell(1,L);
xHatMat{1} = xHat;
xHatMat_l = zeros(sys.N_l,L);
xHatMat_n = zeros(sys.N_n,L);
poseTruth = zeros(sys.N_n, L);
poseTruth(:,1) = poseTruth0;
wMat = zeros(Params.Npart, L);
wMat(:,1) = p0.w*ones(Params.Npart,1);
Neff = zeros(1,L);
Neff(1) = 1/sum(wMat(:,1).^2);
yMat = zeros(sys.N_l,L);
for ii = 2:L
    
    %Propagate Dynamics
    poseTruth(:,ii) = sys.f_n(poseTruth(:,ii-1)) + mvnrnd(zeros(sys.N_n,1),sys.Pnu_n)';
       
    %generate a measurement
    eta = mvnrnd(zeros(sys.N_l,1),sys.Peta)';
    y = sys.h(poseTruth(:,ii)) + sys.C(poseTruth(:,ii))*mapTruth + sys.D(poseTruth(:,ii))*eta;
    
    %run RB Particle Filter
    [xHat, xHat_l, xHat_n] = rbpfSLAM(sys, y, xHat, Params);
    
    %store values
    xHatMat{ii} = xHat;
    xHatMat_l(:,ii) = xHat_l;
    xHatMat_n(:,ii) = xHat_n;
    for kk = 1:Params.Npart
        wMat(kk,ii) = xHat{kk}.w;
    end
    yMat(:,ii) = y;
    
    %calculate effective number of particles
    Neff(ii) = 1/sum(wMat(:,ii).^2);
    
    disp(ii/L)
    
end

%extract map estimate
mapHatXMat = xHatMat_l(1:3:(dim*Nmap - 2),:);
mapHatYMat = xHatMat_l(2:3:(dim*Nmap - 1),:);
mapHatZMat = xHatMat_l(3:3:(dim*Nmap - 0),:);

%mapping error
eMapX = mapHatXMat - mapTruthX;
eMapY = mapHatYMat - mapTruthY;
eMapZ = mapHatZMat - mapTruthZ;
eMap = sqrt(mean(eMapX.^2 + eMapY.^2 + eMapZ.^2,1));

%mapping bias
mapBias = -[mean(eMapX(:,end));
    mean(eMapY(:,end));
    mean(eMapZ(:,end))];


%% Plotting

figure
subplot(3,1,1)
plot(t,xHatMat_n(1,:))
hold on
plot(t,poseTruth(1,:))
plot(t,xHatMat_n(1,:) + mapBias(1))
legend('Est Traj.','True Traj', 'Estimate Plus Bias')
ylabel('x')
title('Localization Performance')
subplot(3,1,2)
plot(t,xHatMat_n(2,:))
hold on
plot(t,poseTruth(2,:))
plot(t,xHatMat_n(2,:) + mapBias(2))
legend('Est Traj.','True Traj', 'Estimate Plus Bias')
ylabel('y')
subplot(3,1,3)
plot(t,xHatMat_n(3,:))
hold on
plot(t,poseTruth(3,:))
plot(t,xHatMat_n(3,:) + mapBias(3))
legend('Est Traj.','True Traj', 'Estimate Plus Bias')
ylabel('z')
xlabel('Time')

figure
subplot(3,1,1)
plot(t,xHatMat_n(4,:))
hold on
plot(t,poseTruth(4,:))
legend('Est Traj.','True Traj')
ylabel('v_x')
title('Velocity Estimation Performance')
subplot(3,1,2)
plot(t,xHatMat_n(5,:))
hold on
plot(t,poseTruth(5,:))
legend('Est Traj.','True Traj')
ylabel('v_y')
subplot(3,1,3)
plot(t,xHatMat_n(6,:))
hold on
plot(t,poseTruth(6,:))
legend('Est Traj.','True Traj')
ylabel('v_z')
xlabel('Time')

figure
subplot(2,1,1)
scatter(mapHatXMat(:,end) - mapTruthX,mapHatYMat(:,end) - mapTruthY,'x');
hold on
scatter(mapHatXMat(:,end) + mapBias(1) - mapTruthX, mapHatYMat(:,end) + mapBias(2) - mapTruthY,5,'filled');
grid on
xlabel('x')
ylabel('y')
title('Mapping Performance')
legend('Map Error','Map Error Plus Bias')
subplot(2,1,2)
scatter(mapHatYMat(:,end) - mapTruthY,mapHatZMat(:,end) - mapTruthZ,'x');
hold on
scatter(mapHatYMat(:,end) + mapBias(2) - mapTruthY, mapHatZMat(:,end) + mapBias(3) - mapTruthZ,5,'filled');
grid on
xlabel('y')
ylabel('z')
legend('Map Error','Map Error Plus Bias')


figure
plot(t,wMat)
xlabel('Time')
ylabel('Particle Weights')

figure
plot(t, Neff)
xlabel('Time')
ylabel('Effective Number of Particles')

figure
scatter3(mapHatXMat(:,end),mapHatYMat(:,end), mapHatZMat(:,end))
hold on
scatter3(mapTruthX,mapTruthY, mapTruthZ)
scatter3(mapHatXMat(:,end)+mapBias(1),...
    mapHatYMat(:,end)+mapBias(2), ...
    mapHatZMat(:,end)+mapBias(3),4,'filled')
xlabel('x')
ylabel('y')
zlabel('z')
title('Mapping Performance')
legend('Estimated Map','True Map','Estimate Plus Bias')

figure
scatter3(eMapX(:,end), eMapY(:,end), eMapZ(:,end))
title('Mapping Error')
xlabel('x')
ylabel('y')
ylabel('z')
lims = abs(axis);
axis([-max(lims(1:2)) max(lims(1:2)) -max(lims(3:4)) max(lims(3:4)) -max(lims(5:6)) max(lims(5:6))]);

%plot velocity of truth compared to all particles at time
timeIdx = L;
figure
hold on
for ii = 1:Npart
    targ = xHatMat{timeIdx}{ii}.xHat_n;
    scatter3(targ(4),targ(5),targ(6),'b')
end
scatter3(poseTruth(4,timeIdx),poseTruth(5,timeIdx),poseTruth(6,timeIdx),'rx')
title('Particle Positions and Truth')

%plot location of first map point in each particle compared to truth at
%time 2
figure
hold on
for ii = 1:Npart
    targ = xHatMat{timeIdx}{ii}.xHat_l;
    scatter3(targ(1),targ(2),targ(3),'b')
end
scatter3(mapTruth(1),mapTruth(2),mapTruth(3),'rx')
title('Particle Velocities and Truth')

%look at location of measurement and all of the predicted measurements at
%time 2
figure
hold on
for ii = 1:Npart
    xHat_n = xHatMat{timeIdx}{ii}.xHat_n;
    xHat_l = xHatMat{timeIdx}{ii}.xHat_l;
    h_n = sys.h(xHat_n);
    C = sys.C(xHat_n);
    targ = h_n + C*xHat_l;
    scatter3(targ(1),targ(2),targ(3),'b');
end
scatter3(yMat(1,timeIdx),yMat(2,timeIdx),yMat(3,timeIdx),'rx')
title('Expected Measurements and Measurement')

% plot the measurement and all the nearby error elipses
figure
hold on
R = sys.Peta;
for ii = 1:Npart
    xHat_n = xHatMat{timeIdx}{ii}.xHat_n;
    xHat_l = xHatMat{timeIdx}{ii}.xHat_l;
    h_n = sys.h(xHat_n);
    C = sys.C(xHat_n);
    D = sys.D(xHat_n);
    P_l = xHat{ii}.P_l;
    targ = h_n + C*xHat_l;
    P = C*P_l*C' + D*R*D';
    plot(targ,'b')
%     e_elipse(gcf,P(1:2,1:2),targ(1:2),1,'r')
    
end
plot(yMat(:,timeIdx), 'r')
title('Expected Measurements and Measurement')  
grid on

%% Functions

%propagation function
function xOut = f_n(x, dt)

Phi = [x(4:6); -.1*x(4:6).^3];

xOut = x + dt*Phi;

end

%measurement function
function y = h(x, Nmap)

%locals
r = x(1:3);

%output
y = repmat(-r,Nmap,1);

end
