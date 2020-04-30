clear
close all
clc

%generate random point cloud
N = 100;
P = rand(3*N,1);

%rotation and translation
Ttrue = [.1 3 .4]';
Etrue = [.1 -.2 .01]';
Rtrue = angle2dcm(Etrue(1),Etrue(2),Etrue(3),'ZYX');

Ptrans = zeros(3*N,1);
for ii = 1:N
    p = P(3*ii-2:3*ii);
    Ptrans(3*ii-2:3*ii) = Rtrue*(p + Ttrue);
end

%cost
cost = @(x)alignCost(x,P,Ptrans);

options = optimoptions('lsqnonlin','MaxFunctionEvaluations',100000, ...
    'MaxIterations',100000);

%calculate alignment
xHat = lsqnonlin(cost,zeros(6,1),[],[],options);
That = xHat(1:3);
Rhat = angle2dcm(xHat(4),xHat(5),xHat(6),'ZYX');

cost(xHat)
cost(zeros(6,1))

truePlot = zeros(3,N);
estPlot = zeros(3,N);
for ii = 1:N
    truePlot(:,ii) = Ptrans(3*ii-2:3*ii);
    estPlot(:,ii) = Rhat*(P(3*ii-2:3*ii) + That);
end

figure
scatter3(truePlot(1,:),truePlot(2,:),truePlot(3,:))
hold on
scatter3(estPlot(1,:),estPlot(2,:),estPlot(3,:),4,'filled')
legend('True','Estimate')