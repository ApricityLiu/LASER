function [acc,nmi,pur,results] = run_f_LASER(folds,f,X,numInst,truthF,lambda1,...
    lambda2,lambda3,dim,r,numClust)
    ind_folds = folds{f};
    for iv = 1:length(X)
        X1 = X{iv}';
        X1 = NormalizeFea(X1,1);
        X2 = X1;

        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        X1(ind_0,:) = 0;    
        Xv{iv} = X1';

        W1 = eye(numInst);
        ind_1 = find(ind_folds(:,iv) == 1);
        W1(ind_1,:) = [];
        Wv{iv} = W1;                            

        Xev{iv} = X2(ind_1,:)';
    end
    clear X1 X2 ind_0 ind_1 W1;
    [~, ~, P, ~, ~, ~, ~,~] = IMVC_weighted_LSR_LatentConstraint(Xv, Xev,...
        Wv, truthF, lambda1, lambda2, lambda3, dim, r);
    [acc,nmi,pur,results] = compute_P_results(P,numClust,truthF);
end

