% Corey Marcus
% Advanced Estimation
% This script runs a monte carlo analysis on a SLAM system using an
% rao-blackwell particle filter

clear
close all
clc

%% Options
N_MC = 25; %number of monte carlo runs
createFirstIterationPlots = true; %create a bunch of nice plots for the first MC run
playFinishedNoise = false; %plays a tone when finished

%% Approximate Covariance for Angular Process Noise
N = 10000; %number of samples
Peuler = 0.00005*eye(3);
muEuler = [0 0 0]';

quatMat = zeros(N,4);
for ii = 1:N
    E = mvnrnd(muEuler,Peuler);
    quatMat(ii,:) = angle2quat(E(1),E(2),E(3),'ZYX');
end

muQuat = mean(quatMat,1);
covQuat = cov(quatMat);

%cleanup
clear quatMat N

%% System
dt = .25;
t = 0:dt:20*dt;
L = length(t);
dim = 3; %dimension of the model
Nstate = 13; %dimension of nonlinear state (pos, vel, quat inertial to body, and rate wrt inertial expressed in body)
Nmap = 10; %number of map objects
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
sys.Pnu_n = blkdiag(.0010*eye(3),.0005*eye(3),covQuat,.0005*eye(3));
sys.Peta = 0.01*eye(dim*Nmap);
sys.N_n = Nstate;
sys.N_l = dim*Nmap;
sys.Peuler = Peuler;

%% RBPF Parameters
Npart = 3000;
Params.Npart = Npart;
Params.estimateAngles = true;

%% Data Storage Intialiation
truePose = zeros(Nstate,L,N_MC);
estPoseRBPF = zeros(Nstate,L,N_MC);
errPoseRBPF = zeros(Nstate - 1,L,N_MC); %look at error in terms of angles
trueMap = zeros(3,Nmap,N_MC);
estMapRBPF = zeros(3,Nmap,L,N_MC);
errMapRBPF = zeros(3,Nmap,L,N_MC);
eNormPose = zeros(L,N_MC);
eNormMap = zeros(L,N_MC);
NeffRBPF = zeros(L,N_MC);
estPoseCovRBPF = zeros(Nstate,Nstate,L,N_MC);
estMapCovRBPF = zeros(3*Nmap,3*Nmap,L,N_MC);

%% Begin Monte Carlo
tic; %starting a timer
invalidRuns = []; %tracking statistics
for ii = 1:N_MC
    %% Truth Initialization
    mapBounds = [-15 15];
    muMap = mapBounds(1) + (mapBounds(2) - mapBounds(1))*rand(dim*Nmap,1);
    Pmap0 = 0.1*eye(dim*Nmap);
    muTheta0 = zeros(3,1);
    Ptheta0 = 5*Peuler;
    mapTruthIter = mvnrnd(muMap, Pmap0)';
    trueMap(1,:,ii) = mapTruthIter(1:3:(dim*Nmap - 2));
    trueMap(2,:,ii) = mapTruthIter(2:3:(dim*Nmap - 1));
    trueMap(3,:,ii) = mapTruthIter(3:3:(dim*Nmap - 0));
    muPose = zeros(Nstate,1);
    muPose(4:6) = [1 1 1]';
    muPose(1:3) = [-10 -10 -10];
    muPose(11:13) = [.1 .1 .1];
    Ppose0 = 5*sys.Pnu_n;
    thetaTruth0 = mvnrnd(muTheta0,Ptheta0)';
    %using a scalar first quaternion representation
    truePose(7:10,1,ii) = angle2quat(thetaTruth0(1),thetaTruth0(2),thetaTruth0(3))';
    truePose([1:6 11:13],1,ii) = mvnrnd(muPose([1:6 11:13]), Ppose0([1:6 11:13], [1:6 11:13]))';
    
    %% Filter Initialization
    
    %create the particle linear estimate and weight
    p0.w = 1/Npart;
    p0.xHat_l = muMap;
    p0.P_l = Pmap0;
    
    %create all the particles
    xHat = cell(Npart,1);
    for jj = 1:Npart
        p0.xHat_n = zeros(Nstate,1);
        p0.xHat_n([1:6 11:13]) = mvnrnd(muPose([1:6 11:13]), Ppose0([1:6 11:13], [1:6 11:13]))';
        
        %initialize attitude
        thetaParticle = mvnrnd(muTheta0, Ptheta0)';
        p0.xHat_n(7:10) = angle2quat(thetaParticle(1),thetaParticle(2),thetaParticle(3))';
        
        %assign
        xHat{jj} = p0;
    end
    
    %effective number of particles
    NeffRBPF(1,ii) = Npart;
    
    %% First Estimate
    xHatMap0 = zeros(dim*Nmap,1);
    for jj = 1:Npart
        xHatMap0 = xHatMap0 + xHat{jj}.w * xHat{jj}.xHat_l;
        estPoseRBPF(:,1,ii) = estPoseRBPF(:,1,ii) + xHat{jj}.w * xHat{jj}.xHat_n;
    end
    
    %normalize quaternion
    estPoseRBPF(7:10,1,ii) = estPoseRBPF(7:10,1,ii)/norm(estPoseRBPF(7:10,1,ii));
    
    %extract map estimate
    estMapRBPF(1,:,1,ii) = xHatMap0(1:dim:(dim*Nmap - 2),:);
    estMapRBPF(2,:,1,ii) = xHatMap0(2:dim:(dim*Nmap - 1),:);
    estMapRBPF(3,:,1,ii) = xHatMap0(3:dim:(dim*Nmap - 0),:);
    
    %get initial MMSE estimate
    xHat_n = zeros(sys.N_n,1);
    for jj = 1:Npart
        xHat_n = xHat_n + xHat{jj}.w*xHat{jj}.xHat_n;
    end
    
    %normalize quaternion
    xHat_n(7:10) = xHat_n(7:10)/norm(xHat_n(7:10));
            
    %initial linear estimate
    xHat_l = muMap;
    
    %Covariance
    for kk = 1:Npart
        estPoseCovRBPF(:,:,1,ii) = estPoseCovRBPF(:,:,1,ii) + ...
            xHat{kk}.w*(xHat{kk}.xHat_n - xHat_n)*(xHat{kk}.xHat_n - xHat_n)';
        estMapCovRBPF(:,:,1,ii) = estMapCovRBPF(:,:,1,ii) + ...
            xHat{kk}.w*(xHat{kk}.P_l + xHat{kk}.xHat_l*xHat{kk}.xHat_l' - ...
            xHat_l*xHat_l');
    end
    
    %% Run Model
    for jj = 2:L
        
        % add process noise
        toProp = truePose(:,jj-1,ii) + mvnrnd(zeros(sys.N_n,1),sys.Pnu_n)';

        %draw random euler angles and create new quaternion
        muQuat = truePose(7:10,jj-1,ii);  
        randEuler = mvnrnd([0 0 0]', sys.Peuler);
        quatNoise = angle2quat(randEuler(1),randEuler(2),randEuler(3),'ZYX');
        toProp(7:10) = quatmultiply(muQuat', quatNoise)';
        
        %Propagate Dynamics
        truePose(:,jj,ii) = sys.f_n(toProp);
        
        %generate a measurement
        eta = mvnrnd(zeros(sys.N_l,1),sys.Peta)';
        y = sys.h(truePose(:,jj,ii)) + sys.C(truePose(:,jj,ii))*mapTruthIter + sys.D(truePose(:,jj,ii))*eta;
        
        %run RB Particle Filter
        [xHat, xHat_l, xHat_n] = rbpfSLAM(sys, y, xHat, Params, muQuat);
        
        %store values
        estPoseRBPF(:,jj,ii) = xHat_n;
        estMapRBPF(1,:,jj,ii) = xHat_l(1:dim:(dim*Nmap - 2),:);
        estMapRBPF(2,:,jj,ii) = xHat_l(2:dim:(dim*Nmap - 1),:);
        estMapRBPF(3,:,jj,ii) = xHat_l(3:dim:(dim*Nmap - 0),:);
        
        %effective number of particles
        wMat = zeros(1,Npart);
        for kk = 1:Npart
            wMat(kk) = xHat{kk}.w;
        end
        NeffRBPF(jj,ii) = 1/sum(wMat.^2);
        
        %Covariance
        for kk = 1:Npart
            estPoseCovRBPF(:,:,jj,ii) = estPoseCovRBPF(:,:,jj,ii) + ...
                xHat{kk}.w*(xHat{kk}.xHat_n - xHat_n)*(xHat{kk}.xHat_n - xHat_n)';
            estMapCovRBPF(:,:,jj,ii) = estMapCovRBPF(:,:,jj,ii) + ...
                xHat{kk}.w*(xHat{kk}.P_l + xHat{kk}.xHat_l*xHat{kk}.xHat_l' - ...
                xHat_l*xHat_l');
        end
        
        
        %Output Diagnostics
        clc
        fprintf('MC Iteration: %i / %i \n',ii,N_MC)
        fprintf('Time Step: %i / %i \n',jj,L)
        
        %output estimated completion time
        timeElapsed = toc;
        fracComplete = ((ii - 1)*L + jj)/(L*N_MC);
        estTimeRemaining = (1 - fracComplete)*timeElapsed/fracComplete;
        fprintf('Time Elapsed (min): %4.2f \n',timeElapsed/60)
        fprintf('Estimated Time Remaining (min): %4.2f \n',estTimeRemaining/60)
        fprintf('Number of Potentially Invalid Runs Detected: %4i \n',length(invalidRuns))
        
        
    end
    
    %% Align Truth and Estimate for a more fair comparison
    estMapVect = xHat_l;
    
    %cost function and options
    cost = @(x)alignCost(x,estMapVect,mapTruthIter);
    options = optimoptions('lsqnonlin','MaxFunctionEvaluations',100000, ...
        'MaxIterations',100000,'Algorithm','levenberg-marquardt','Display','off');
    
    
    %check to see if cost returns NaN

    if(isnan(cost(zeros(6,1))))
        invalidRuns = [invalidRuns ii];
        disp('Warning: Invalid Run Detected')
        continue;
    end
    
    %calculate alignment
    xHat = lsqnonlin(cost,zeros(6,1),[],[],options);
    That = xHat(1:3);
    Rhat = angle2dcm(xHat(4),xHat(5),xHat(6),'ZYX');
    
    %align all of the quantities
    for jj = 1:L
        
        %rotate and translate position
        estPoseRBPF(1:3,jj,ii) = Rhat*(estPoseRBPF(1:3,jj,ii) + That);
        
        %rotate velocity
        estPoseRBPF(4:6,jj,ii) = Rhat*estPoseRBPF(4:6,jj,ii);
        
        %align pose
        q = dcm2quat(Rhat);
        estPoseRBPF(7:10,jj,ii) = quatmultiply(q,estPoseRBPF(7:10,jj,ii)')';
        
        %angular rate is unchanged since it is with respect to the body
        
        %align the map
        for kk = 1:Nmap
            estMapRBPF(:,kk,jj,ii) = Rhat*(estMapRBPF(:,kk,jj,ii) + That);
        end
        
    end
    
    %% Calculate Error
    for jj = 1:L
        
        %calculate error euler angles
        eQuat = quatmultiply(quatconj(truePose(7:10,jj,ii)'),estPoseRBPF(7:10,jj,ii)');
        [eX, eY, eZ] = quat2angle(eQuat);
        
        %calculate error in the pose
        errPoseRBPF(1:6,jj,ii) = estPoseRBPF(1:6,jj,ii) - truePose(1:6,jj,ii);
        errPoseRBPF(7:9,jj,ii) = [eX, eY, eZ]';
        errPoseRBPF(10:12,jj,ii) = estPoseRBPF(11:13,jj,ii) - truePose(11:13,jj,ii);
        
        %calculate error in the map
        errMapRBPF(:,:,jj,ii) =  estMapRBPF(:,:,jj,ii) - trueMap(:,:,ii);
    end
    
    
end

%look for invalid runs
if(~isempty(invalidRuns))
    disp('Warning: The following runs may be invalid:')
    disp(invalidRuns)
    disp('Recommend changing idxs variable')
end

%% Calculate some statistics

%you need to redefine this manually if there are invalid runs
idxs = 1:N_MC;

%Get the average covariance from the filters
estPoseCovRBPFmean = mean(estPoseCovRBPF,4);
estMapCovRBPFmean = mean(estMapCovRBPF,4);
estMapCovRBPFmeanX = zeros(1,L);
estMapCovRBPFmeanY = zeros(1,L);
estMapCovRBPFmeanZ = zeros(1,L);

%get the mean error
errPoseRBPFmean = mean(errPoseRBPF,3);
errMapRBPFmean = mean(errMapRBPF,4);

%get the sample error covariance
errPoseCovRBPFsample = zeros(Nstate - 1,Nstate - 1,L);
errMapCovRBPFsample = zeros(3*Nmap,3*Nmap,L);
errMapCovRBPFsampleX = zeros(1,L);
errMapCovRBPFsampleY = zeros(1,L);
errMapCovRBPFsampleZ = zeros(1,L);
for ii = 1:L
    
    %reshape the map error for covariance analysis
    targ = squeeze(errMapRBPF(:,:,ii,:));
    targ = reshape(targ,[3*Nmap, N_MC]);
    
    errPoseCovRBPFsample(:,:,ii) = cov(squeeze(errPoseRBPF(:,ii,:))');
    errMapCovRBPFsample(:,:,ii) = cov(targ');
    
    estMapCovRBPFmeanX(ii) = mean(diag(estMapCovRBPF(1:3:(3*Nmap - 2), 1:3:(3*Nmap - 2), ii)));
    estMapCovRBPFmeanY(ii) = mean(diag(estMapCovRBPF(2:3:(3*Nmap - 1), 2:3:(3*Nmap - 1), ii)));
    estMapCovRBPFmeanZ(ii) = mean(diag(estMapCovRBPF(3:3:(3*Nmap - 0), 3:3:(3*Nmap - 0), ii)));
    
    errMapCovRBPFsampleX(ii) = mean(diag(errMapCovRBPFsample(1:3:(3*Nmap - 2), 1:3:(3*Nmap - 2), ii)));
    errMapCovRBPFsampleY(ii) = mean(diag(errMapCovRBPFsample(2:3:(3*Nmap - 1), 2:3:(3*Nmap - 1), ii)));
    errMapCovRBPFsampleZ(ii) = mean(diag(errMapCovRBPFsample(3:3:(3*Nmap - 0), 3:3:(3*Nmap - 0), ii)));
end

%% Plotting For First MC Run
if(createFirstIterationPlots)
    
    %target MC index
    targMC = 1;
    
    %create some nice vectors for plotting
    vMatTruth = zeros(3,L);
    vMatEst = zeros(3,L);
    v = [1 0 0]';
    for ii = 1:L
        vMatTruth(:,ii) = quatrotate(truePose(7:10,ii,targMC)',v')';
        vMatEst(:,ii) = quatrotate(estPoseRBPF(7:10,ii,targMC)',v')';
    end
    
    figure
    plot3(estPoseRBPF(1,:,targMC), estPoseRBPF(2,:,targMC), estPoseRBPF(3,:,targMC))
    hold on
    plot3(truePose(1,:,targMC), truePose(2,:,targMC), truePose(3,:,targMC))
    quiver3(estPoseRBPF(1,:,targMC), estPoseRBPF(2,:,targMC), estPoseRBPF(3,:,targMC),...
        vMatEst(1,:), vMatEst(2,:), vMatEst(3,:))
    quiver3(truePose(1,:,targMC), truePose(2,:,targMC), truePose(3,:,targMC),...
        vMatTruth(1,:), vMatTruth(2,:), vMatTruth(3,:))
    legend('Est Traj.','True Traj','Est Pointing', 'True Pointing')
    xlabel('x')
    ylabel('y')
    zlabel('z')
    title('Localization Performance For First MC Run')
    
end

%% Plotting MC Statistics

%plot the position error for every MC run
figure
for ii = idxs
    
    %x
    subplot(3,1,1)
    hold on
    plot(t, errPoseRBPF(1,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
    %y
    subplot(3,1,2)
    hold on
    plot(t, errPoseRBPF(2,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
    %z
    subplot(3,1,3)
    hold on
    plot(t, errPoseRBPF(3,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
end
subplot(3,1,1)
title('Position Error for all Runs')
ylabel('e_x')
plot(t, squeeze(sqrt(estPoseCovRBPFmean(1,1,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(1,1,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(1,1,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(1,1,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(1,:), 'r')
legend('Filter 1 \sigma', 'Sample 1 \sigma', 'Mean Error')

subplot(3,1,2)
ylabel('e_y')
plot(t, squeeze(sqrt(estPoseCovRBPFmean(2,2,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(2,2,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(2,2,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(2,2,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(2,:), 'r')
legend('Filter 1 \sigma', 'Sample 1 \sigma', 'Mean Error')

subplot(3,1,3)
xlabel('Time (sec)')
ylabel('e_z')
plot(t, squeeze(sqrt(estPoseCovRBPFmean(3,3,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(3,3,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(3,3,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(3,3,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(3,:), 'r')
legend('Filter 1 \sigma', 'Sample 1 \sigma', 'Mean Error')


%plot the velocity error for every MC run
figure
for ii = idxs
    
    %x
    subplot(3,1,1)
    hold on
    plot(t, errPoseRBPF(4,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
    %y
    subplot(3,1,2)
    hold on
    plot(t, errPoseRBPF(5,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
    %z
    subplot(3,1,3)
    hold on
    plot(t, errPoseRBPF(6,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
end

subplot(3,1,1)
title('Velocity Error for all Runs')
ylabel('e_x')
idx = 4;
plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')
legend('Filter 1 \sigma', 'Sample 1 \sigma', 'Mean Error')

subplot(3,1,2)
ylabel('e_y')
idx = 5;
plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')

subplot(3,1,3)
xlabel('Time (sec)')
ylabel('e_z')
idx = 6;
plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')

%plot the angle error for every MC run
figure
for ii = idxs
    
    %x
    subplot(3,1,1)
    hold on
    plot(t, errPoseRBPF(7,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
    %y
    subplot(3,1,2)
    hold on
    plot(t, errPoseRBPF(8,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
    %z
    subplot(3,1,3)
    hold on
    plot(t, errPoseRBPF(9,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
end

subplot(3,1,1)
title('Error Euler Angles for all Runs')
ylabel('e_x')
idx = 7;
% plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k')
% plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')
legend('Sample 1 \sigma', 'Mean Error')

subplot(3,1,2)
ylabel('e_y')
idx = 8;
% plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k')
% plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')
legend('Sample 1 \sigma', 'Mean Error')

subplot(3,1,3)
xlabel('Time (sec)')
ylabel('e_z')
idx = 9;
% plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k')
% plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx,idx,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')
legend('Sample 1 \sigma', 'Mean Error')

%plot the body rate error for every MC run
figure
for ii = idxs
    
    %x
    subplot(3,1,1)
    hold on
    plot(t, errPoseRBPF(10,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
    %y
    subplot(3,1,2)
    hold on
    plot(t, errPoseRBPF(11,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
    %z
    subplot(3,1,3)
    hold on
    plot(t, errPoseRBPF(12,:,ii), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
    
end

subplot(3,1,1)
title('Body Rate Error for all Runs')
ylabel('e_x')
idx = 10;
plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx+1,idx+1,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx+1,idx+1,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')

subplot(3,1,2)
ylabel('e_y')
idx = 11;
plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx+1,idx+1,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx+1,idx+1,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')

subplot(3,1,3)
xlabel('Time (sec)')
ylabel('e_z')
idx = 12;
plot(t, squeeze(sqrt(estPoseCovRBPFmean(idx+1,idx+1,:))), '-k')
plot(t, -squeeze(sqrt(estPoseCovRBPFmean(idx+1,idx+1,:))), '-k', 'HandleVisibility', 'off')
plot(t, squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b')
plot(t, -squeeze(sqrt(errPoseCovRBPFsample(idx,idx,:))), '--b', 'HandleVisibility', 'off')
plot(t, errPoseRBPFmean(idx,:), 'r')

% Plot the Mapping Error for Every MC Run and Every Point
figure
for ii = idxs
    for jj = 1:Nmap
        
        %x
        subplot(3,1,1)
        hold on
        plot(t, squeeze(errMapRBPF(1,jj,:,ii)), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
        
        %y
        subplot(3,1,2)
        hold on
        plot(t, squeeze(errMapRBPF(2,jj,:,ii)), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
        
        %z
        subplot(3,1,3)
        hold on
        plot(t, squeeze(errMapRBPF(3,jj,:,ii)), 'Color', [.8 .8 .8],'HandleVisibility', 'off')
        
    end
end

subplot(3,1,1)
title('Mapping Error for all Runs and Map Points')
ylabel('e_x')
idx = 1;
plot(t, sqrt(estMapCovRBPFmeanX), '-k')
plot(t, -sqrt(estMapCovRBPFmeanX), '-k', 'HandleVisibility', 'off')
plot(t, sqrt(errMapCovRBPFsampleX), '--b')
plot(t, -sqrt(errMapCovRBPFsampleX), '--b', 'HandleVisibility', 'off')
plot(t, squeeze(mean(errMapRBPFmean(idx,:,:),2)), 'r')
legend('Filter 1 \sigma', 'Sample 1 \sigma', 'Mean Error')

subplot(3,1,2)
ylabel('e_y')
idx = 2;
plot(t, sqrt(estMapCovRBPFmeanY), '-k')
plot(t, -sqrt(estMapCovRBPFmeanY), '-k', 'HandleVisibility', 'off')
plot(t, sqrt(errMapCovRBPFsampleY), '--b')
plot(t, -sqrt(errMapCovRBPFsampleY), '--b', 'HandleVisibility', 'off')
plot(t, squeeze(mean(errMapRBPFmean(idx,:,:),2)), 'r')
legend('Filter 1 \sigma', 'Sample 1 \sigma', 'Mean Error')

subplot(3,1,3)
xlabel('Time (sec)')
ylabel('e_z')
idx = 3;
plot(t, sqrt(estMapCovRBPFmeanZ), '-k')
plot(t, -sqrt(estMapCovRBPFmeanZ), '-k', 'HandleVisibility', 'off')
plot(t, sqrt(errMapCovRBPFsampleZ), '--b')
plot(t, -sqrt(errMapCovRBPFsampleZ), '--b', 'HandleVisibility', 'off')
plot(t, squeeze(mean(errMapRBPFmean(idx,:,:),2)), 'r')
legend('Filter 1 \sigma', 'Sample 1 \sigma', 'Mean Error')

% plot the effective number of particles
figure
hold on
for ii = idxs
    plot(t, NeffRBPF(:,ii), 'Color', [.8 .8 .8])
end
xlabel('Time (sec)')
ylabel('Neff')
title('Effective Number of Particles for all Runs')


%% Play Alarm
if(playFinishedNoise)
    load gong
    sound(y,Fs)
end

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

