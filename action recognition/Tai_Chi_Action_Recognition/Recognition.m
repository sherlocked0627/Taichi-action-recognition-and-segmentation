function accuracy = Recognition(traj,label,ratio,k)
% accuracy: accuracy of action recognition
%
% traj：store trajectory data, size(m*n)
% label：labels
% ratio：the ratio of training data and test data

num = 50;  
leftshoulder = 45; 
rightshoulder = 17; 
lefthand = 1;      
n = size(traj,2);     % number of actions

vector = cell(1,n);  
for i = 1:n
    A = traj{lefthand,i};        
    A1 = points_filter(A,num);  
    A2 = traj_turn(A1,traj,leftshoulder,rightshoulder,i);
    vector(1,i) = MDL(A2);
end

D = cell2mat(vector);   
Num = 64;
[means, covariances, priors,LL, POSTERIORS] = gmm_para(D,Num);
data = traj(lefthand,:);
encoding = zeros(Num*6,n);
for j = 1:n
encoding(:,j) = vl_fisher(data{1,j}, means, covariances, priors);
end

accuracy = traj_svm(encoding,label,ratio,k); 
end


    
    




