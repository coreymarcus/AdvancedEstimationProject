% Corey Marcus
% Advanced Estimation
% This script runs a rbpf on a very simple nonlinear model.

clear
close all
clc

%% System
dt = .1;
t = 0:dt:10;
L = length(t);
Nstate = 4; %dimension of full state
Nstate_l = 2;
Nstate_n = Nstate - Nstate_l;
sys.f_n = @(x_n) [-x_n(1)*cos(x_n(2)) + x_n(1);
    -x_n(2)*sin(x_n(1)) + x_n(2)];
sys.A_n = @(x_n) eye(2);
sys.B_n = @(x_n) eye(Nstate_n);
sys.f_l = 0;
sys.A_l = @(x_n) [1 dt;
    0 1];
sys.B_l = @(x_n) eye(Nstate_l);
sys.h = @(x_n) [x_n(1) + x_n(2);
    x_n(2)];
sys.C = @(x_n) [0 0;
    1 0];
sys.D = @(x_n) eye(2);
sys.Pnu_n = .01*eye(Nstate_n);
sys.Pnu_l = .01*eye(Nstate_l);
sys.Peta = .01*eye(2);
sys.N_n = Nstate_n;
sys.N_l = Nstate_l;

%% Filter Parameters
Params.Npart = 200;


%% Truth Initialization
muState0 = [0 0 0 0]';
Pstate0 = eye(Nstate);
state0 = mvnrnd(muState0',Pstate0)';

%% Filter Initialization

%create the particle linear estimate and weight
p0.w = 1/Params.Npart;
p0.xHat_l = muState0(1:2);
p0.P_l = Pstate0(1:2,1:2);

%create all the particles
xHat = cell(Params.Npart,1);
for ii = 1:Params.Npart
    p0.xHat_n = mvnrnd(muState0(3:4), Pstate0(3:4,3:4))';
    xHat{ii} = p0;
end

%% Run
xHatMat = cell(1,L);
xHatMat_l = zeros(sys.N_l,L);
xHatMat_n = zeros(sys.N_n,L);
stateTruth = zeros(Nstate, L);
stateTruth(:,1) = state0;
wMat = zeros(Params.Npart, L);
wMat(:,1) = p0.w*ones(Params.Npart,1);
Neff = zeros(1,L);
Neff(1) = 1/sum(wMat(:,1).^2);
for ii = 2:L
    
    %Propagate Dynamics
    stateTruth(1:2,ii) = sys.A_l(stateTruth(3:4,ii-1))*stateTruth(1:2,ii-1) ...
        + mvnrnd(zeros(sys.N_l,1),sys.Pnu_l)';
    stateTruth(3:4,ii) = sys.f_n(stateTruth(3:4,ii-1)) ...
        + sys.A_n(stateTruth(3:4,ii-1))*stateTruth(1:2,ii-1)...
        + mvnrnd(zeros(sys.N_n,1),sys.Pnu_n)';  
    
    %generate a measurement
    eta = mvnrnd(zeros(sys.N_l,1),sys.Peta)';
    y = sys.h(stateTruth(3:4,ii)) + sys.C(stateTruth(3:4,ii))*stateTruth(1:2,ii) + sys.D(stateTruth(3:4,ii))*eta;
    
    %run RB Particle Filter
    [xHat, xHat_l, xHat_n] = rbpf(sys, y, xHat, Params);
    
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

%% Plotting

figure
subplot(4,1,1)
% plot(t,xHatMat_n(1,:))
hold on
plot(t,stateTruth(1,:))
ylabel('x1')
legend('Est Traj.','True Traj')
subplot(4,1,2)
% plot(t,xHatMat_n(2,:))
hold on
plot(t,stateTruth(2,:))
ylabel('x2')
subplot(4,1,3)
% plot(t,xHatMat_n(3,:))
hold on
plot(t,stateTruth(3,:))
ylabel('x3')
legend('Est Traj.','True Traj')
subplot(4,1,4)
% plot(t,xHatMat_n(4,:))
hold on
plot(t,stateTruth(4,:))
ylabel('x4')
xlabel('Time')
legend('Est Traj.','True Traj')

% figure
% plot(t,wMat)
% xlabel('Time')
% ylabel('Particle Weights')
% 
% figure
% plot(t, Neff)
% xlabel('Time')
% ylabel('Effective Number of Particles')
% 
% figure
% scatter(mapHatXMat(:,end),mapHatYMat(:,end))
% hold on
% scatter(mapTruthX,mapTruthY)
% xlabel('x')
% ylabel('y')
% title('Mapping Performance')
% legend('Estimated Map','True Map')
