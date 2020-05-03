function [xHatOut, xMMSE_l, xMMSE_n] = rbpfSLAM(sys, y, xHat, Params)
%rbpf - Rao Blackwellised Particle Filter Optimized for SLAM - provides an
%   iteration of an RBPF given a system, measurement, and turning
%   parameters. Source is Zanetti's notes on RBPFs. Calculations
%   optimized for SLAM (i.e. Map Assumed Static)
%
% Inputs
% sys - system structure with the following elements
%   f_n = nonlinear state propagation function handle
%   B_n = function handle to generate matrix mapping noise to nonlinear
%       state
%   h = function hangle mapping nonlinear state to measurement
%   C = function handle creating matrix to map linear state to measurement
%   D = function handle creating matrix to map noise to measurement
%   Pnu_n = Covariance for nonlinear propagation noise
%   Peta = Covariance for measurement noise
%   N_n = dim of nonlinear state
%   N_l = dim of linear state
% y = the measurement at time t = k
% xHat - the [Npart x 1] state estimate cell array where each cell is a
%   structure corresponding to a particle with the following fields
%   w - the particle weight
%   xHat_l - an estimate of the linear states
%   P_l - the covariance of the linear states
%   xHat_n - an estimate of the nonlinear states
%   mapPts - a vector of idxs for map points associated with the particle
% Params - parameter structure with the following parameters
%   Npart = Number of particles to be used for the nonlinear states
%   estimateAngles = bool with logic for attitude estimation
%   Nmap = number of map points
%   Nassociate = number of map points associated with each particle
%
% Outputs
% xHatOut = the state estimate at time t = k, cell array where each cell is a
%   structure corresponding to a particle with the following fields
%   w - the particle weight
%   xHat_l - an estimate of the linear states
%   P_l - the covariance of the linear states


%% Setup

%local variables
Npart = Params.Npart;
estimateAngles = Params.estimateAngles;
Nmap = Params.Nmap;
Nassociate = Params.Nassociate;


%% Resampling

%get all the weights
wMat = zeros(Npart,1);
for ii = 1:Npart
    wMat(ii) = xHat{ii}.w;
end

%split particles
split = sysresample(wMat);

%duplicate particles
xHat = xHat(split);

%reassign weights and associated map points
for ii = 1:Npart
    xHat{ii}.w = 1/Npart;
    seq = randperm(Nmap);
    xHat{ii}.mapPts = seq(1:Nassociate);
end

%% Propagation

for ii = 1:Npart
    
    %local variables
    xHat_n_minus = xHat{ii}.xHat_n;
    B_n = sys.B_n(xHat_n_minus);
    
    % Draw Particles From Importance Distrubution (Bootstrap - p(x_k|x_k-1)
    mu = sys.f_n(xHat_n_minus);
    xHat{ii}.xHat_n = mvnrnd(mu, B_n*sys.Pnu_n*B_n')';
    
    if(estimateAngles)
        mu_q = mu(7:10);
        %draw random euler angles and create new quaternion
        randEuler = mvnrnd([0 0 0]', sys.Peuler);
        quatNoise = angle2quat(randEuler(1),randEuler(2),randEuler(3),'ZYX');
        xHat{ii}.xHat_n(7:10) = quatmultiply(quatNoise, mu_q');
    end
    
    
end

%% Update

%track weights for normalization
% logwPrimeMat = zeros(Npart,1);

% we use this scale factor to prevent underflow errors
a = 2;

for ii = 1:Npart
    
    % Local variables
    xHat_n = xHat{ii}.xHat_n;
    xHat_l = xHat{ii}.xHat_l;
    mapPts = xHat{ii}.mapPts;
    
    % target indicies
    targIdx = zeros(3*Nassociate,1);
    for jj = 1:Nassociate
        targIdx(3*jj - 2) = 3*mapPts(jj) - 2;
        targIdx(3*jj - 1) = 3*mapPts(jj) - 1;
        targIdx(3*jj - 0) = 3*mapPts(jj) - 0;
    end
    
    % down sample to only get target indicies
    R = sys.Peta(targIdx,targIdx);
    yTarg = y(targIdx);
    xHat_l = xHat_l(targIdx);
    h_n = sys.h(xHat_n);
    h_n = h_n(targIdx);
    C = sys.C(xHat_n);
    C = C(targIdx,targIdx);
    D = sys.D(xHat_n);
    D = D(targIdx,targIdx);
    P_l = xHat{ii}.P_l;
    P_l = P_l(targIdx,targIdx);
    
    % Evauluate the gaussian and update weight
    p = gaussEval(yTarg, h_n + C*xHat_l, a*(C*P_l*C' + D*R*D'));
    wMat(ii) = p;
    
    
end

% wPrimePrime = exp(logwPrimeMat - max(logwPrimeMat));

% Renormalize the weights, raise to the correct power, then normalize again
wMat = wMat/sum(wMat);
Nattrition = sum(wMat == 0);
fprintf('Lost Points Before Exponent: %4i \n',Nattrition);
wMat = wMat.^a;
wTrack = sum(wMat);
wMat = wMat/sum(wMat);
Neff = 1/sum(wMat.^2);
Nattrition = sum(wMat == 0);
fprintf('Lost Points After Exponent: %4i \n',Nattrition);
fprintf('Effective Number of Points: %4f \n',Neff);

if(wTrack == 0)
    disp('Error: Unstable Weight Normalization!')
end

%% Resample 2

%split particles
split = sysresample(wMat);

%resample particles
xHat = xHat(split);

%reassign weights and associated map points
for ii = 1:Npart
    xHat{ii}.w = 1/Npart;
    seq = randperm(Nmap);
    xHat{ii}.mapPts = seq(1:Nassociate);
end

%resample
b = 1; %resample tightness

for ii = 1:Npart
    
    % Local variables
    xHat_n = xHat{ii}.xHat_n;
    xHat_l = xHat{ii}.xHat_l;
    mapPts = xHat{ii}.mapPts;
    w_minus = xHat{ii}.w;
    
    % Draw Particles
    mu = xHat_n;
    mu_q = mu(7:10);
    xHat{ii}.xHat_n = mvnrnd(mu, b*(sys.Pnu_n))';
    
    %draw random euler angles and create new quaternion
    randEuler = mvnrnd([0 0 0]',b*(sys.Peuler));
    quatNoise = angle2quat(randEuler(1),randEuler(2),randEuler(3),'ZYX');
    xHat{ii}.xHat_n(7:10) = quatmultiply(mu_q', quatNoise);
    
%     %recalculate locals
%     xHat_n = xHat{ii}.xHat_n;
%     h_n = sys.h(xHat_n);
%     C = sys.C(xHat_n);
%     D = sys.D(xHat_n);
%     P_l = xHat{ii}.P_l;
%     w_minus = 1/Npart;
    
    % target indicies
    targIdx = zeros(3*Nassociate,1);
    for jj = 1:Nassociate
        targIdx(3*jj - 2) = 3*mapPts(jj) - 2;
        targIdx(3*jj - 1) = 3*mapPts(jj) - 1;
        targIdx(3*jj - 0) = 3*mapPts(jj) - 0;
    end
    
    % down sample to only get target indicies
    R = sys.Peta(targIdx,targIdx);
    yTarg = y(targIdx);
    xHat_l = xHat_l(targIdx);
    h_n = sys.h(xHat_n);
    h_n = h_n(targIdx);
    C = sys.C(xHat_n);
    C = C(targIdx,targIdx);
    D = sys.D(xHat_n);
    D = D(targIdx,targIdx);
    P_l = xHat{ii}.P_l;
    P_l = P_l(targIdx,targIdx);
    
    % Evauluate the gaussian and update weight
    p = gaussEval(yTarg, h_n + C*xHat_l, a*(C*P_l*C' + D*R*D'));
    wMat(ii) = w_minus*p;
    
    %     % Update the KFs with the measurement
    %     W = C*P_l*C' + D*R*D';
    %     K = P_l*C'/W;
    %     xHat{ii}.xHat_l = xHat_l + K*(y - h_n - C*xHat_l);
    %     xHat{ii}.P_l = P_l - K*W*K';
    
end

% Renormalize the weights
wMat = wMat/sum(wMat);
wMat = wMat.^a;
wMat = wMat/sum(wMat);
Neff = 1/sum(wMat.^2);
Nattrition = sum(wMat == 0);
fprintf('Lost Points After Exponent: %4i \n',Nattrition);
fprintf('Effective Number of Points: %4f \n',Neff);

%reassign weights
for ii = 1:Npart
    xHat{ii}.w = wMat(ii);
end

%% Resample 3

%split particles
split = sysresample(wMat);

%resample particles
xHat = xHat(split);

%reassign weights and associated map points
for ii = 1:Npart
    xHat{ii}.w = 1/Npart;
    seq = randperm(Nmap);
%     xHat{ii}.mapPts = seq(1:Nassociate);
end

%resample
b = 1; %resample tightness

for ii = 1:Npart
    
    % Local variables
    xHat_n = xHat{ii}.xHat_n;
    xHat_l = xHat{ii}.xHat_l;
    mapPts = xHat{ii}.mapPts;
    w_minus = xHat{ii}.w;
    
    % Draw Particles
    mu = xHat_n;
    mu_q = mu(7:10);
    xHat{ii}.xHat_n = mvnrnd(mu, b*(sys.Pnu_n))';
    
    %draw random euler angles and create new quaternion
    randEuler = mvnrnd([0 0 0]',b*(sys.Peuler));
    quatNoise = angle2quat(randEuler(1),randEuler(2),randEuler(3),'ZYX');
    xHat{ii}.xHat_n(7:10) = quatmultiply(mu_q', quatNoise);
    
%     %recalculate locals
%     xHat_n = xHat{ii}.xHat_n;
%     h_n = sys.h(xHat_n);
%     C = sys.C(xHat_n);
%     D = sys.D(xHat_n);
%     P_l = xHat{ii}.P_l;
%     w_minus = 1/Npart;
    
    % target indicies
    targIdx = zeros(3*Nassociate,1);
    for jj = 1:Nassociate
        targIdx(3*jj - 2) = 3*mapPts(jj) - 2;
        targIdx(3*jj - 1) = 3*mapPts(jj) - 1;
        targIdx(3*jj - 0) = 3*mapPts(jj) - 0;
    end
    
    % down sample to only get target indicies
    R = sys.Peta(targIdx,targIdx);
    yTarg = y(targIdx);
    xHat_l = xHat_l(targIdx);
    h_n = sys.h(xHat_n);
    h_n = h_n(targIdx);
    C = sys.C(xHat_n);
    C = C(targIdx,targIdx);
    D = sys.D(xHat_n);
    D = D(targIdx,targIdx);
    P_l = xHat{ii}.P_l;
    P_l = P_l(targIdx,targIdx);
    
    % Evauluate the gaussian and update weight
    p = gaussEval(yTarg, h_n + C*xHat_l, a*(C*P_l*C' + D*R*D'));
    wMat(ii) = w_minus*p;
    
    %     % Update the KFs with the measurement
    %     W = C*P_l*C' + D*R*D';
    %     K = P_l*C'/W;
    %     xHat{ii}.xHat_l = xHat_l + K*(y - h_n - C*xHat_l);
    %     xHat{ii}.P_l = P_l - K*W*K';
    
end

% Renormalize the weights
wMat = wMat/sum(wMat);
wMat = wMat.^a;
wMat = wMat/sum(wMat);
Neff = 1/sum(wMat.^2);
Nattrition = sum(wMat == 0);
fprintf('Lost Points After Exponent: %4i \n',Nattrition);
fprintf('Effective Number of Points: %4f \n',Neff);

%reassign weights
for ii = 1:Npart
    xHat{ii}.w = wMat(ii);
end

%% Output


%find the MMSE Estimates
xMMSE_l = zeros(sys.N_l,1);
xMMSE_n = zeros(sys.N_n,1);

for ii = 1:Npart
    xMMSE_l = xMMSE_l + xHat{ii}.w*xHat{ii}.xHat_l;
    xMMSE_n = xMMSE_n + xHat{ii}.w*xHat{ii}.xHat_n;
end

if(estimateAngles)
    %normalize quaternion
    xMMSE_n(7:10) = xMMSE_n(7:10)/norm(xMMSE_n(7:10));
end


xHatOut = xHat;
end

