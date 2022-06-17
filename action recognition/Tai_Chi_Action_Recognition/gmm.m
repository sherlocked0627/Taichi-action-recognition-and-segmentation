

function [means, covariances, priors,LL, POSTERIORS] = gmm_para(D,Num)

numClusters = Num;

[IDX,IDX1,IDX2] = vl_kmeans(D,numClusters,'NumRepetitions',10);    %利用kmeans聚类为高斯聚类设置初值
sumd = ones(3,64);
W = zeros(1,64);
for ii = 1:nozeronumber(D);
    for kk = 1:numClusters
        if IDX1(1,ii) == kk;
            W(1,kk) = W(1,kk) + 1;
        end
    end
end

[means, covariances, priors,LL, POSTERIORS] = vl_gmm(D, numClusters,'InitMeans',IDX,'InitCovariances',sumd,'InitPriors',W,'NumRepetitions',10);
end


