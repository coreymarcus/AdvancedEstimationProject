clear
close all
clc


q = ones(10,1);
q(2) = 20;
q(8) = 50;
q = q/norm(q);
qRes = sysresample(q)