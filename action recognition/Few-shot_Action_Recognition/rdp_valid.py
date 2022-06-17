from math import sqrt
import numpy as np
import scipy.io as scio
from operator import index
import numpy as np
import scipy.io as scio
from sklearn.metrics import classification_report
import copy
from numpy.linalg import det
from minisom import MiniSom
import os
from sklearn import svm
from sklearn.metrics import accuracy_score

def pointfilter(a):
    j_num = a.shape[0]
    length = a.shape[2]
    for j in range(0,j_num):
        for m in range(2,(length-2)):
            a[j,:,m] = (a[j,:,m-2]+a[j,:,m-1]+a[j,:,m]+a[j,:,m+1]+a[j,:,m+2])/5
    B = a[:,:,2:length-2]
    return B

def distance(a, b):
    return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def point_line_distance(point, start, end):
    if (start[0] == end[0] and start[1] == end[1] and start[2] == end[2]):
        return distance(point, start)
    else:
        x1 = [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
        x2 = [start[0] - point[0], start[1] - point[1], start[2] - point[2]]
        n = np.linalg.norm(np.cross(x1,x2))
        d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2 + (end[2] - start[2]) ** 2)
        return n / d

def rdp(points, epsilon):
    """Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, points.shape[1] - 1):
        d = point_line_distance(points[:,i], points[:,0], points[:,-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        results = rdp(points[:,:index+1], epsilon)[:-1] + rdp(points[:,index:], epsilon)
    else:
        results = [points[:,0], points[:,-1]]

    return results

def closest(mylist, number):
    answer = []
    for i in mylist:
        answer.append(abs(number-i))
    return answer.index(min(answer))

def rdp_frame(num, a, c):
    index = []
    length = []
    for epl in np.arange(0.1, 10, 0.1):
        m = 'epl_{a}'
        mm = m.format(a=format(epl, '.2f'))
        n = '/home/xuleiyang/workspace/TaiChi/action/rdp_data/a{a}_t{c}.npy'
        nn = n.format(a=a, c=c)
        aa = np.load(nn, allow_pickle=True).item()
        index1 = aa[mm]
        index.append(index1)
        length.append(len(index1))
    close_index = closest(length, num)
    index_choose = index[close_index]
    return index_choose
        
def points2circle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    num1 = len(p1)
    num2 = len(p2)
    num3 = len(p3)
    if (num1 == num2) and (num2 == num3):
        if num1 == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)
        elif num1 != 3:
            return None
    else:
        return None

    temp01 = p1 - p2
    temp02 = p3 - p2
    temp03 = np.cross(temp01, temp02)
    temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02)
    if temp < 10**-6:
        return [0,0,0],0

    temp1 = np.vstack((p1, p2, p3))
    temp2 = np.ones(3).reshape(3, 1)
    mat1 = np.hstack((temp1, temp2))  # size = 3x4

    m = +det(mat1[:, 1:])
    n = -det(np.delete(mat1, 1, axis=1))
    p = +det(np.delete(mat1, 2, axis=1))
    q = -det(temp1)

    temp3 = np.array([p1 @ p1, p2 @ p2, p3 @ p3]).reshape(3, 1)
    temp4 = np.hstack((temp3, mat1))
    temp5 = np.array([2 * q, -m, -n, -p, 0])
    mat2 = np.vstack((temp4, temp5))  # size = 4x5

    A = +det(mat2[:, 1:])
    B = -det(np.delete(mat2, 1, axis=1))
    C = +det(np.delete(mat2, 2, axis=1))
    D = -det(np.delete(mat2, 3, axis=1))
    E = +det(mat2[:, :-1])

    pc = -np.array([B, C, D]) / 2 / A
    r = np.sqrt(B * B + C * C + D * D - 4 * A * E) / 2 / abs(A)

    return pc, 1/r

def cos_sim(a,b):
    a = np.mat(a)
    b = np.mat(b)
    num = float(a * b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        cos = 0
    else:
        cos = num/denom
    sim = 0.5 + 0.5 * cos
    return sim

def train_sample_feature(clear, A, feature_number):
    X_train = np.zeros([len(clear)-1, feature_number])
    for i in range(0, len(clear)-1):
        a = clear[i]
        b = clear[i+1]
        action = A[:,a:b]
        start = copy.copy(action[:,0])
        for j in range(0, action.shape[1]):
            action[0,j] = action[0,j] - start[0]
            action[1,j] = action[1,j] - start[1]
            action[2,j] = action[2,j] - start[2]
        X_train[i,0:3] = action[:,-1]
        len_fea = 0
        for m in range(0, action.shape[1]-1):
            x1 = action[0,m]
            y1 = action[1,m]
            z1 = action[2,m]
            x2 = action[0,m+1]
            y2 = action[1,m+1]
            z2 = action[2,m+1]
            len_fea = len_fea + np.sqrt(np.square(x2-x1)+np.square(y2-y1)+np.square(z2-z1))
        X_train[i,3] = len_fea
        cos_x = cos_sim(action[:,-1], [1,0,0])
        X_train[i,4] = cos_x
        cos_y = cos_sim(action[:,-1], [0,1,0])
        X_train[i,5] = cos_y
        cos_z = cos_sim(action[:,-1], [0,0,1])
        X_train[i,6] = cos_z
    return X_train

def som_train(x_train, som_shape, max_iter, sigma, learning_rate):
    M = x_train.shape[1]
    # som = MiniSom(size, size, M, sigma=1, learning_rate=0.5)
    som = MiniSom(som_shape[0], som_shape[1], M, sigma, learning_rate)
    som.train_batch(x_train, max_iter, verbose=True)
    winner_coordinates = np.array([som.winner(x) for x in x_train]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    return som, cluster_index

def classify(som, data, sample, label):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = som.labels_map(sample, label)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def arr_size(arr, size):
    s = []
    count = 0
    for i in (size):
        s.append(arr[count:count+i])
        count = count+i
    return s

def DTW(a,b):
    distance = 0
    aa = a.reshape([-1,10])
    bb = b.reshape([-1,10])
    for i in range(0, aa.shape[0]):
        aa_cut = aa[i,:]
        bb_cut = bb[i,:]
        distance = distance + dtw(aa_cut, bb_cut)
    
    return distance

def dtw(a, b):
    an = a.size
    bn = b.size
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0
    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = dist(a[ai],b[bi]) + minimum_cost
    dis = cumdist[an, bn]
    return dis

def dist(a,b):
    # distance = abs(b-a)
    # return distance
    if a == b:
        distance = 0
    else:
        distance = 1
    return distance

def geometry(left_shoulder, right_shoulder, right_elbow, hand):
    geo = []
    geo2 = []
    geo3 = []
    geo4 = []
    geo5 = []
    geo6 = []
    geo7 = []
    for i in range(hand.shape[0]-1):
        x0 = [1,0,0]
        x1 = left_shoulder[i,:] - right_shoulder[i,:]
        x2 = right_shoulder[i,:] - right_elbow[i,:]
        x3 = hand[i,:] - right_elbow[i,:]
        x1_ = left_shoulder[i+1,:] - right_shoulder[i+1,:]
        x2_ = right_shoulder[i+1,:] - right_elbow[i+1,:]
        x3_ = hand[i+1,:] - right_elbow[i+1,:]
        cos2 = cos_sim(x2,x0)
        cos3 = cos_sim(x3,x0)
        cos4 = cos_sim(x1,x2)
        cos5 = cos_sim(x2,x3)
        cos6 = cos_sim(x1_,x2_)
        cos7 = cos_sim(x2_,x3_)
        geo2.append(cos2)
        geo3.append(cos3)
        geo4.append(cos4)
        geo5.append(cos5)
        geo6.append(cos6 - cos4)
        geo7.append(cos7 - cos5)
    geo= geo2 + geo3 + geo4 + geo5 + geo6 + geo7
    return geo

def TRAIN(joint_index, inter_num, choose, som_shape, max_iter, sigma, learning_rate):
    feature_number = 7
    x_train = np.zeros([1,feature_number])
    y_train = []
    length = []
    train_geo = []
    ###### Read Data
    for a in range(1, 11):
        for c in choose:
            dataFile = '/home/xuleiyang/workspace/TaiChi/action/action.npy'
            s = 'a{m}'
            aa = np.load(dataFile, allow_pickle=True).item()
            bb = aa[s.format(m=a)][c]
            action = bb[joint_index]
            ##### Key Frames
            frame_index = rdp_frame(inter_num, a, c)
            frame_index_geo = np.rint(np.linspace(0,action.shape[1]-1,inter_num)).astype(int)
            ##### Geometry features
            cc = copy.copy(bb)
            left_shoulder = cc[44][:,frame_index_geo].T
            right_shoulder = cc[16][:,frame_index_geo].T
            right_elbow = cc[17][:,frame_index_geo].T
            hand = cc[joint_index][:,frame_index_geo].T
            geo = geometry(left_shoulder, right_shoulder, right_elbow, hand)
            train_geo.append(geo)
            #### Train Data
            train_sample = train_sample_feature(frame_index, action, feature_number)
            x_train = np.vstack((x_train, train_sample))
            length.append(len(train_sample))
            y_train.append(a)
    x_train = np.delete(x_train, 0, axis=0)
    som, cluster_index = som_train(x_train, som_shape, max_iter, sigma, learning_rate)
    return som, cluster_index, x_train, y_train, length, train_geo
    
def TEST(joint_index, inter_num, choose, som, sample, label):
    feature_number = 7
    x_test = np.zeros([1,feature_number])
    y_test = []
    length = []
    test_geo = []
    for a in range(1,11):
        for c in choose:
            dataFile = '/home/xuleiyang/workspace/TaiChi/action/action.npy'
            s = 'a{m}'
            aa = np.load(dataFile, allow_pickle=True).item()
            bb = aa[s.format(m=a)][c]
            action = bb[joint_index]
            frame_index = rdp_frame(inter_num, a, c)
            frame_index_geo = np.rint(np.linspace(0,action.shape[1]-1,inter_num)).astype(int)
            cc = copy.copy(bb)
            left_shoulder = cc[44][:,frame_index_geo].T
            right_shoulder = cc[16][:,frame_index_geo].T
            right_elbow = cc[17][:,frame_index_geo].T
            hand = cc[joint_index][:,frame_index_geo].T
            geo = geometry(left_shoulder, right_shoulder, right_elbow, hand)
            test_geo.append(geo)
            train_sample = train_sample_feature(frame_index, action, feature_number)
            x_test = np.vstack((x_test, train_sample))
            length.append(len(train_sample))
            y_test.append(a)
    x_test = np.delete(x_test, 0, axis=0)
    cluster_index = classify(som, x_test, sample, label)
    return cluster_index, y_test, length, test_geo


##### Train Data
# size = math.ceil(np.sqrt(5 * np.sqrt(2160)))
inter_num = 16
size = 4
som_shape = (size, size)
max_iter = 8000
sigma = 1
learning_rate = 0.5
joint_index = 19
som_rh, X_train_rh, x_train_rh, y_train, length_train_rh, train_geo = TRAIN(joint_index, inter_num, [0], som_shape, max_iter, sigma, learning_rate)
X_train = arr_size(X_train_rh, length_train_rh)


##### Test Data
# X_test_rh, y_test, length_test_rh, test_geo = TEST(joint_index, inter_num, [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19], som_rh, x_train_rh, X_train_rh)
X_test_rh, y_test, length_test_rh, test_geo = TEST(joint_index, inter_num, [10], som_rh, x_train_rh, X_train_rh)
X_test = arr_size(X_test_rh, length_test_rh)

##### SVM
clf = svm.SVC(C=2, kernel='rbf', gamma=1, decision_function_shape='ovr') 
clf.fit(train_geo, y_train)
test_label = clf.predict(test_geo)
acc = clf.score(test_geo, y_test)
print(acc)
dis_svm = clf.decision_function(test_geo)
for i in range (len(dis_svm)):
    max_value = dis_svm[i].max()
    dis_svm[i] = max_value - dis_svm[i]
    pass

##### DTW
acc = []
for cc in np.arange(0.8, 1.5, 0.01):
    class_num = 1
    y_pred = np.zeros([1,len(X_test)])
    for h in range(0, len(X_test)):
        distance = np.zeros([1,10])
        for e in range(0,1):
            test_sample = np.array(X_test[h])
            for i in range(0,10):
                train_sample = X_train[i*class_num:(i+1)*class_num]
                for j in range(0, len(train_sample)):
                    distance[e,i] = distance[e,i] + dtw(train_sample[j], test_sample)
                distance[e,i] = distance[e,i] / class_num
        distance_final = np.sum(distance, axis = 0)
        distance_final = np.sum(distance, axis = 0) + dis_svm[h]*cc
        y_pred[0,h] = np.argmin(distance_final) + 1
        
    accuracy = accuracy_score(y_test, y_pred[0])
    acc.append(accuracy)

print(acc)

pass




