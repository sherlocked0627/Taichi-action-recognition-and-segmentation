function accuracy = traj_svm(encoding,label,ratio,k)

traindata = cell(1,k);
trainlabel = cell(1,k);
testdata = cell(1,k);
testlabel = cell(1,k);
for i =1:k
    [x,y] = find(label==i);
    num = length(y);
    encoding_middle = encoding(:,y);
    label_middle = label(1,y);
    p = round(num*ratio);
    r1 = randperm(size(encoding_middle,2));
    encoding__middle = encoding_middle(:,r1);
    label__middle = label_middle(1,r1);
    traindata(1,i) = encoding__middle(:,1:p);
    trainlabel(1,i) = label__middle(1,1:p);
    testdata(1,i) = encoding__middle(:,p+1:num);
    testlabel(1,i) = label__middle(:,p+1:num);
end
train_data = cell2mat(traindata);  
train_label = cell2mat(trainlabel);
test_data = cell2mat(testdata);  
test_label = cell2mat(testlabel); 

rr = randperm(size(test_data,2));
test_data = test_data(:,rr);
test_label = test_label(:,rr);

model = svmtrain(train_label',train_data','-c 20 -g 0.09');

[predict_label,accuracy,dec_values] = svmpredict(test_label',test_data',model);
end

    
    
