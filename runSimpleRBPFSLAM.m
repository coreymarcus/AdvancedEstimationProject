% Corey Marcus
% Advanced Estimation
% This script runs a rbpf on a very simple SLAM model. Intended for testing.

clear
close all
clc

%% System
dt = .1;
t = 0:dt:10;
L = length(t);
dim = 2; %dimension of the model
Nstate = 2*dim; %dimension of nonlinear state (2D pos and vel)
Nmap = 50; %number of map objects
Acam = eye(Nstate);
Acam(1,3) = dt;
Acam(2,4) = dt;
sys.f_n = @(x_n) Acam*x_n;
sys.B_n = @(x_n) eye(Nstate);
sys.A_l = @(x_n) eye(dim*Nmap);
sys.B_l = @(x_n) eye(dim*Nmap);
sys.h = @(x_n) repmat(-x_n(1:(Nstate/2),1),Nmap,1);
sys.C = @(x_n) eye(dim*Nmap);
sys.D = @(x_n) eye(dim*Nmap);
sys.Pnu_n = .01*eye(Nstate);
sys.Peta = .01*eye(dim*Nmap); %map assumed static
sys.N_n = Nstate;
sys.N_l = dim*Nmap;

%% Filter Parameters
Params.Npart = 50;


%% Truth Initialization
mapBounds = [-10 10];
muMap = mapBounds(1) + (mapBounds(2) - mapBounds(1))*rand(2*Nmap,1);
Pmap0 = eye(2*Nmap);
mapTruth = mvnrnd(muMap, Pmap0)';
mapTruthX = mapTruth(1:2:(2*Nmap - 1));
mapTruthY = mapTruth(2:2:(2*Nmap));
muPose = zeros(Nstate,1);
muPose(3:4) = [1 1]';
Ppose0 = .1*eye(Nstate);
poseTruth0 = mvnrnd(muPose,Ppose0)';

%% Filter Initialization

%create the particle linear estimate and weight
p0.w = 1/Params.Npart;
p0.xHat_l = muMap;
p0.P_l = Pmap0;

%create all the particles
xHat = cell(Params.Npart,1);
for ii = 1:Params.Npart
    p0.xHat_n = mvnrnd(muPose, Ppose0)';
    xHat{ii} = p0;
end

%% Run
xHatMat = cell(1,L);
xHatMat_l = zeros(sys.N_l,L);
xHatMat_n = zeros(sys.N_n,L);
poseTruth = zeros(sys.N_n, L);
poseTruth(:,1) = poseTruth0;
wMat = zeros(Params.Npart, L);
wMat(:,1) = p0.w*ones(Params.Npart,1);
Neff = zeros(1,L);
Neff(1) = 1/sum(wMat(:,1).^2);
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
    
    %calculate effective number of particles
    Neff(ii) = 1/sum(wMat(:,ii).^2);
    
end

%extract map estimate
mapHatXMat = xHatMat_l(1:2:(2*Nmap - 1),:);
mapHatYMat = xHatMat_l(2:2:(2*Nmap),:);

%mapping error
eMapX = mapHatXMat - mapTruthX;
eMapY = mapHatYMat - mapTruthY;
eMap = sqrt(mean(eMapX.^2 + eMapY.^2,1));

%% Plotting

figure
subplot(4,1,1)
plot(t,xHatMat_n(1,:))
hold on
plot(t,poseTruth(1,:))
ylabel('x1')
legend('Est Traj.','True Traj')
subplot(4,1,2)
plot(t,xHatMat_n(2,:))
hold on
plot(t,poseTruth(2,:))
ylabel('x2')
subplot(4,1,3)
plot(t,xHatMat_n(3,:))
hold on
plot(t,poseTruth(3,:))
ylabel('x3')
legend('Est Traj.','True Traj')
subplot(4,1,4)
plot(t,xHatMat_n(4,:))
hold on
plot(t,poseTruth(4,:))
ylabel('x4')
xlabel('Time')
legend('Est Traj.','True Traj')

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