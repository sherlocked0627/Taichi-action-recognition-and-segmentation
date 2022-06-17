function accuracy = Recognition(traj,label,ratio,k)
% accuracy：识别准确率
%
% traj：m*n的cell数组，储存m个点的n个动作轨迹数据（左手手腕）
% label：1*n动作对应的类别1~k
% ratio：训练样本和测试样本比例

num = 50;  %滤波次数
leftshoulder = 45;  %左肩点数据在traj中的位置
rightshoulder = 17;  %右肩点数据在traj中的位置
lefthand = 1;       %左手点数据在traj中的位置
n = size(traj,2);     %动作数量

vector = cell(1,n);    %储存轨迹向量
for i = 1:n
    A = traj{lefthand,i};        %提取一个动作轨迹
    A1 = points_filter(A,num);     %轨迹中值滤波
    A2 = traj_turn(A1,traj,leftshoulder,rightshoulder,i);     %轨迹旋转
    vector(1,i) = MDL(A2);       %轨迹分割
end

D = cell2mat(vector);   
Num = 64;      %混合高斯模型聚类数量
[means, covariances, priors,LL, POSTERIORS] = gmm_para(D,Num);    %计算混合高斯模型参数
data = traj(lefthand,:);
encoding = zeros(Num*6,n);
for j = 1:n
encoding(:,j) = vl_fisher(data{1,j}, means, covariances, priors);   %计算Fisher向量
end

accuracy = traj_svm(encoding,label,ratio,k);      %利用支持向量机对动作进行分类
end


    
    




