function [ACC,NMI,Pur,results] = compute_P_results(P,numClust,truth)
%COMPUTE_P_RESULTS 此处显示有关此函数的摘要
%   此处显示详细说明

P(isnan(P)) = 0;
P(isinf(P)) = 1e5;
new_F = P';
norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
% avoid divide by zero
for i = 1:size(norm_mat,1)
    if (norm_mat(i,1)==0)
        norm_mat(i,:) = 1;
    end
end
new_F = new_F./norm_mat; 
rand('seed',230);
pre_labels = kmeans(real(new_F),numClust,'emptyaction','singleton','replicates',20,'display','off');
result = ClusteringMeasure(truth,pre_labels);
ACC = result(1)*100;
NMI = result(2)*100;
Pur = result(3)*100;
results = [ACC,NMI,Pur];
end

