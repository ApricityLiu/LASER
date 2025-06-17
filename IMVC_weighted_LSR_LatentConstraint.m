function [Zv, Uv, P, S, J, alphav, obj, iter] = IMVC_weighted_LSR_LatentConstraint(Xv, Xev, ...
    Wv, truth, lambda1, lambda2, lambda3, dim, r)
%IMVC_weighted_LSR
%Min \sum_{v} alphav^r ( ||Xv+Xev*Zv*Wv-Uv*P||^2_F + lambda1*||Zv||_F^2 ) +
%lambda2*||P-P*S||^2_F + lambda3*||S||_*
% s.t., Uv'*Uv=I, \sum_{v} alphav = 1, alphav>=0

%Min \sum_{v} alphav^r ( ||Xv+Xev*Zv*Wv-Uv*P||^2_F + lambda1*||Zv||_F^2 ) +
%lambda2*||P-P*S||^2_F + lambda3*||J||_* + < Gamma2, J-S> + 0.5*||J-S||^2_F
% s.t., Uv'*Uv=I, \sum_{v} alphav = 1, alphav>=0

mu = 1e-4;
rho = 1.2;
max_mu = 1e6;
tol = 1e-3;

viewnum = length(Xv);
samnum = length(truth);
clustnum = length(unique(truth));
for v=1:viewnum
   missnum(v) = size(Wv{v},1);
   existnum(v) = size(Xev{v},2);
   viewdimnum(v) = size(Xv{v},1);
end

maxiter = 100;

alphav = ones(viewnum,1)/viewnum;
alphav_r = alphav.^r;
PPP = zeros(dim,samnum);
for v=1:viewnum
    rand('seed',v*100);
    linshi_U = rand(viewdimnum(v),dim);
    if viewdimnum(v) > dim
        Uv{v} = orth(linshi_U);
    else
        Uv{v} = (orth(linshi_U'))';
    end
    
    Zv{v} = rand(existnum(v),missnum(v));
    
    PPP = PPP + Uv{v}'*(Xv{v}+Xev{v}*Zv{v}*Wv{v});

end
P = PPP/viewnum;

S = zeros(samnum);
J = zeros(samnum);
Gamma2 = zeros(samnum);

for iter = 1:maxiter
   %---------------P-----------------%
   temp1 = zeros(dim,samnum);
   for v=1:viewnum
      temp1 = temp1 + alphav_r(v)*Uv{v}'*(Xv{v}+Xev{v}*Zv{v}*Wv{v}); 
   end
   P = temp1*inv( sum(alphav_r)*eye(samnum) + lambda2*eye(samnum) - lambda2*S - lambda2*S' + lambda2*(S*S') );
   P(isnan(P)) = 0;
   P(isinf(P)) = 1e5;
   %--------------S-------------------%
   S = inv(2*lambda2*(P'*P)+mu*eye(samnum));
   S = S*(2*lambda2*(P'*P)+Gamma2+mu*J);
   %--------------J-------------------%
   J = sigma_soft_thresh(S-Gamma2/mu,lambda3/mu);
    %--------------Uv Zv ----------------%
    NormX = 0;
    for v=1:viewnum
        Zv{v} = inv(Xev{v}'*Xev{v}+lambda1*eye(existnum(v)));
        Zv{v} = Zv{v}*(Xev{v}'*(Uv{v}*P-Xv{v})*Wv{v}');
        
        linshi = Xv{v}+Xev{v}*Zv{v}*Wv{v};
        temp3 = linshi*P';
        temp3(isnan(temp3)) = 0;
        temp3(isinf(temp3)) = 1e10;
        [Gs,~,Vs] = svd(temp3,'econ');
        Gs(isnan(Gs)) = 0;
        Vs(isnan(Vs)) = 0;
        Uv{v} = Gs*Vs';
        clear Gs Vs;
        
        Rec_error(v) = norm(Xv{v}+Xev{v}*Zv{v}*Wv{v}-Uv{v}*P,'fro')^2 + lambda1*norm(Zv{v},'fro')^2;
        NormX = NormX + norm(Xv{v},'fro')^2;     
    end
    %-------------alphav---------------------------%
    H = bsxfun(@power, Rec_error,1/(1-r));
    alphav = bsxfun(@rdivide,H,sum(H));
    alphav_r = alphav.^r;
    %-------------Gamma2 mu------------------------%
    Gamma2 = Gamma2 + mu*(J-S);
    mu = min(rho*mu,max_mu);
    %------------compute error----------------------%
    errJS = max(max(abs(J-S)));
    obj(iter) = (alphav_r*Rec_error'+lambda2*norm(P-P*S,'fro')^2+lambda3*sum(svd(S)))/NormX;
    fprintf('iter=%d, errJS=%f, Obj=%f\n',iter,errJS,obj(iter));
    
    %----------converge or not--------------------%
    if  errJS< tol && iter > 2 && abs(obj(iter)-obj(iter-1)) < tol
        break;
    end
    
end
end

