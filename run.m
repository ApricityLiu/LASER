
clear;
clc;
close all;

load bbcsport4vbigRnSp.mat;
load bbcsport4vbigRnSp_percentDel_0.1.mat;

fmin = 1;
fmax = 5;

folds = folds(1:5);
    
num_view = length(X);
numClust = length(unique(truth));
numInst  = length(truth);

truthF = truth;
clear truth;

% dim调参范围 10:10:100; 
% lambda1,lambda2,lambda3的调参范围：[1e-5;1e-4;1e-3;1e-2;1e-1;1e0;1e1;1e2;1e3;1e4;1e5];

r = 3;
dim = 10;
lambda1 = 1;
lambda2 = 1e4;
lambda3 = 1e4;

accf = zeros(1,5); nmif = zeros(1,5); purf = zeros(1,5);
 parfor f = fmin:fmax 
     [accf(f),nmif(f),purf(f),~] = run_f_LASER(folds,f,X,numInst,truthF,lambda1,lambda2,lambda3,dim,r,numClust);
 end
 
fprintf('Optimal: dim=%d, lambda1=%f, lambda2=%f, lambda3=%f\n',dim,lambda1,lambda2,lambda3);
fprintf('ACC=%f(%f), NMI=%f(%f), PUR=%f(%f)',mean(accf),std(accf),...
    mean(nmif),std(nmif),mean(purf),std(purf));



