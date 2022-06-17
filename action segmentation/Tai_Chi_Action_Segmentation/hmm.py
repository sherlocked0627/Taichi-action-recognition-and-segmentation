import numpy as np
from hmmlearn import hmm
from metric import f1
import random

def preprocessing(label, train_data, train_label):
    index = np.where(train_label == label)[0].tolist()
    lengths1 = []
    for i in index:
        if i == index[0]:
            X1 = train_data[i,:,:]
            lengths1.append(train_data[i,:,:].shape[1])
        else:
            X1 = np.concatenate([X1, train_data[i,:,:]], axis=1)
            lengths1.append(train_data[i,:,:].shape[1])
    X1 = np.transpose(X1, (1,0))
    return X1, lengths1

seed = 1538574472
random.seed(seed)

train_data = np.load('./dataset/TaiChi/train/data.npy', allow_pickle=True)
train_data = train_data.reshape(train_data.shape[0], -1, train_data.shape[3])
train_label = np.load('./dataset/TaiChi/train/label.npy', allow_pickle=True)

num_components = 3
print('HMM, 3')
### model 1
X1, lengths1 = preprocessing(0, train_data, train_label)
model1 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model1.fit(X1, lengths1)

### model 2
X2, lengths2 = preprocessing(1, train_data, train_label)
model2 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model2.fit(X2, lengths2)

### model 3
X3, lengths3 = preprocessing(2, train_data, train_label)
model3 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model3.fit(X3, lengths3)

### model 4
X4, lengths4 = preprocessing(3, train_data, train_label)
model4 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model4.fit(X4, lengths4)

### model 5
X5, lengths5 = preprocessing(4, train_data, train_label)
model5 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model5.fit(X5, lengths5)

### model 6
X6, lengths6 = preprocessing(5, train_data, train_label)
model6 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model6.fit(X6, lengths6)

### model 7
X7, lengths7 = preprocessing(6, train_data, train_label)
model7 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model7.fit(X7, lengths7)

### model 8
X8, lengths8 = preprocessing(7, train_data, train_label)
model8 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model8.fit(X8, lengths8)

### model 9
X9, lengths9 = preprocessing(8, train_data, train_label)
model9 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model9.fit(X9, lengths9)

### model 10
X10, lengths10 = preprocessing(9, train_data, train_label)
model10 = hmm.GaussianHMM(n_components=num_components, covariance_type="full", n_iter=100)
model10.fit(X10, lengths10)


test_data = np.load('./dataset/TaiChi/test/data.npy', allow_pickle=True)
test_data = test_data.reshape(test_data.shape[0], -1, test_data.shape[3])
test_label = np.load('./dataset/TaiChi/test/label.npy', allow_pickle=True)

predict = []

for j in range(0, test_data.shape[0]):
    sequence = test_data[j,:,:]
    sequence = np.transpose(sequence, (1,0))
    length = [sequence.shape[0]]
    score = []
    score.append(model1.score(sequence, length))
    score.append(model2.score(sequence, length))
    score.append(model3.score(sequence, length))
    score.append(model4.score(sequence, length))
    score.append(model5.score(sequence, length))
    score.append(model6.score(sequence, length))
    score.append(model7.score(sequence, length))
    score.append(model8.score(sequence, length))
    score.append(model9.score(sequence, length))
    score.append(model10.score(sequence, length))
    predict.append(score.index(max(score)))


f1(test_label.tolist(), predict, [0.1,0.25,0.5])

np.savetxt('hmm_pre.txt', np.array(predict[639:1299]))

pass