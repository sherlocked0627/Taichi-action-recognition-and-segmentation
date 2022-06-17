import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from metric import f1
import random

seed = 1538574472
random.seed(seed)

train_data = np.load('./dataset/TaiChi/train/data.npy', allow_pickle=True)
train_data = train_data.reshape(train_data.shape[0], -1)
train_label = np.load('./dataset/TaiChi/train/label.npy', allow_pickle=True)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(train_data, train_label)

test_data = np.load('./dataset/TaiChi/test/data.npy', allow_pickle=True)
test_data = test_data.reshape(test_data.shape[0], -1)
test_label = np.load('./dataset/TaiChi/test/label.npy', allow_pickle=True)

predict = clf.predict(test_data)

acc = accuracy_score(test_label, predict)
print(acc)
f1(test_label.tolist(), predict.tolist(), [0.1,0.25,0.5])
# np.savetxt('st_pre.txt', predict[639:1299])

pass