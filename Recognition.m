function accuracy = Recognition(traj,label,ratio,k)
% accuracy��ʶ��׼ȷ��
%
% traj��m*n��cell���飬����m�����n�������켣���ݣ���������
% label��1*n������Ӧ�����1~k
% ratio��ѵ�������Ͳ�����������

num = 50;  %�˲�����
leftshoulder = 45;  %����������traj�е�λ��
rightshoulder = 17;  %�Ҽ��������traj�е�λ��
lefthand = 1;       %���ֵ�������traj�е�λ��
n = size(traj,2);     %��������

vector = cell(1,n);    %����켣����
for i = 1:n
    A = traj{lefthand,i};        %��ȡһ�������켣
    A1 = points_filter(A,num);     %�켣��ֵ�˲�
    A2 = traj_turn(A1,traj,leftshoulder,rightshoulder,i);     %�켣��ת
    vector(1,i) = MDL(A2);       %�켣�ָ�
end

D = cell2mat(vector);   
Num = 64;      %��ϸ�˹ģ�;�������
[means, covariances, priors,LL, POSTERIORS] = gmm_para(D,Num);    %�����ϸ�˹ģ�Ͳ���
data = traj(lefthand,:);
encoding = zeros(Num*6,n);
for j = 1:n
encoding(:,j) = vl_fisher(data{1,j}, means, covariances, priors);   %����Fisher����
end

accuracy = traj_svm(encoding,label,ratio,k);      %����֧���������Զ������з���
end


    
    




