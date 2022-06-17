from numpy.linalg import det
import scipy.io as scio
import numpy as np
import copy
import csv
from minisom import MiniSom
from math import ceil

def pointfilter(num, a):
    length = a.shape[1]
    for i in range(0,num):
        a = a[:,0:length-4*i]
        for j in range(2,(a.shape[1]-2)):
            a[:,j-2] = (a[:,j-2]+a[:,j-1]+a[:,j]+a[:,j+1]+a[:,j+2])/5
    B = a[:,0:a.shape[1]-4]
    return B

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
            print('error')
            return None
    else:
        print('error')
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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx+1

def curve_primitives(curve):
    label = np.zeros(curve.shape[0])
    for i in range (0, curve.shape[0]):
        if curve[i] > 0.5:
            label[i] = 4
        elif curve[i] < 0.5 and curve[i] > 0.25:
            label[i] = 3
        elif curve[i] < 0.25 and curve[i] > 0.1:
            label[i] = 2
        elif curve[i] < 0.1 and curve[i] > 0:
            label[i] = 1
    return label

def delete_repeat(label):
    new_label = [label[0]]
    start = label[0]
    for i in range (0,label.shape[0]):
        if label[i] != start:
            new_label.append(label[i])
            start = label[i]
    return new_label

def list_in(list,anotherList):
    position = []
    for i in range(0,len(anotherList)):
        if(list[0]==anotherList[i]):
            if(len(anotherList[i:]) >= len(list)):
                c=0
                for j in range(0,len(list)):
                    if(list[j]==anotherList[j+i]):
                        c += 1
                        if(c==len(list)):
                            position.append(i)
                    else:
                        continue
    return position

def cut(k,a_thre,start):
    cut_1 = []
    for j in range(0,k.shape[0]-1):
        if (k[j]>a_thre and k[j+1]<a_thre) or (k[j]<a_thre and k[j+1]>a_thre):
            dis1 = abs(k[j]-a_thre)
            dis2 = abs(k[j+1]-a_thre)
            if cut_1 == []:
                if dis1 > dis2:
                    cut_1.append((j+1)*5+start)
                if dis1 <= dis2:
                    cut_1.append(j*5+start)
            if cut_1 != []:
                if dis1 <= dis2 and cut_1[-1] != j*5+start:
                    cut_1.append(j*5+start)
                if dis1 <= dis2 and cut_1[-1] == j*5+start:
                    cut_1.append((j+1)*5+start)
                if dis1 > dis2:
                    cut_1.append((j+1)*5+start)
            
    return cut_1

def merge(cut,max_cutlen,inter,min_cutlen,start,end):
    index = []
    for i in range(0, len(cut)-1):
        if cut[i+1] - cut[i] <= min_cutlen:
            index.append(i+1)
    for ii in index:
        cut[ii] = 0
    while 0 in cut:
        cut.remove(0)
    
    new_cut = [start]
    distance1 = cut[0] - start
    if distance1 < max_cutlen:
            new_cut.append(cut[0])
    elif distance1 >= max_cutlen:
        add_num = round(distance1/inter)
        add = round(distance1/(add_num-1))
        for b in range(1,add_num-1):
            new_cut.append(start+b*add)
        new_cut.append(cut[0])
    for j in range(0, len(cut)-1):
        dis = cut[j+1] - cut[j]
        if dis < max_cutlen:
            new_cut.append(cut[j+1])
        elif dis >= max_cutlen:
            add_num = round(dis/inter)
            add = round(dis/(add_num-1))
            for a in range(1,add_num-1):
                new_cut.append(cut[j]+a*add)
            new_cut.append(cut[j+1])
    distance2 = end - cut[-1]
    if distance2 < max_cutlen:
            new_cut.append(end)
    elif distance2 >= max_cutlen:
        add_num = round(distance2/inter)
        add = round(distance2/(add_num-1))
        for c in range(1,add_num-1):
            new_cut.append(cut[-1]+c*add)
        new_cut.append(end)
    return new_cut

def min_len(clear):
    index = 0
    min = clear[1] - clear[0]
    for a in range (1,len(clear)-1):
        inter = clear[a+1] - clear[a]
        if inter < min:
            min = inter
            index = a
    return index, min

def takespread(sequence, num):
    length = sequence.shape[1]
    new_sequence = np.zeros([3, num])
    for i in range(num):
        new_sequence[:,i] = sequence[:,int(ceil(i * length / num))]
    return new_sequence

def train_sample(clear, A, min_inter):
    X_train = np.zeros([len(clear)-1, (min_inter)*3])
    for i in range(0, len(clear)-1):
        a = clear[i]
        b = clear[i+1]
        action = A[:,a:b]
        start = copy.copy(action[:,0])
        for j in range(0, action.shape[1]):
            action[0,j] = action[0,j] - start[0]
            action[1,j] = action[1,j] - start[1]
            action[2,j] = action[2,j] - start[2]
        train = takespread(action, min_inter)
        feature = train.flatten(order="F")
        X_train[i,:] = feature.T
    return X_train

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
        k = np.zeros(action.shape[1]-2)
        for n in range(0, action.shape[1]-2):
            pc, k[n] = points2circle(action[:,n].T, action[:,n+1].T, action[:,n+2].T)
        curve_fea = np.mean(k)
        X_train[i,0] = curve_fea
        len_fea = 0
        for m in range(0, action.shape[1]-1):
            x1 = action[0,m]
            y1 = action[1,m]
            z1 = action[2,m]
            x2 = action[0,m+1]
            y2 = action[1,m+1]
            z2 = action[2,m+1]
            len_fea = len_fea + np.sqrt(np.square(x2-x1)+np.square(y2-y1)+np.square(z2-z1))
        X_train[i,1] = len_fea
        direction_fea = action[:,-1] - action[:,-2]
        X_train[i,2:5] = direction_fea
        final_coord = action[:,-1]
        X_train[i,5:8] = final_coord
        average_coord = np.mean(action, axis = 1)
        X_train[i,8:11] = average_coord
    return X_train

def h_dtw(s1,s2):
    r, c = len(s1), len(s2)
    D0 = np.zeros((r+1,c+1))
    D0[0,1:] = np.inf
    D0[1:,0] = np.inf
    D1 = D0[1:,1:]
    for i in range(r):
        for j in range(c):
            if s1[i] == s2[j]:
                D1[i,j] = 0
            else:
                D1[i,j] = 1
    M = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i,j] += min(D0[i,j],D0[i,j+1],D0[i+1,j])

    i,j = np.array(D0.shape) - 2
    p,q = [i],[j]
    while(i>0 or j>0):
        tb = np.argmin((D0[i,j],D0[i,j+1],D0[i+1,j]))
        if tb==0 :
            i-=1
            j-=1
        elif tb==1 :
            i-=1
        else:
            j-=1
        p.insert(0,i)
        q.insert(0,j)
    return D1[-1,-1]

def threepoints_flat(p1, p2, p3):
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    x2 = p2[0]
    y2 = p2[1]
    z2 = p2[2]
    x3 = p3[0]
    y3 = p3[1]
    z3 = p3[2]
    a = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
    b = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
    c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    d = -a*x1-b*y1-c*z1
    return a, b, c, d

def vetor_angle(x1,x2,x3):
    x = x2 - x1
    y = x2 - x3
    Lx = np.sqrt(x[0]**2+x[1]**2+x[2]**2)
    Ly = np.sqrt(y[0]**2+y[1]**2+y[2]**2)
    cos_angle = x.dot(y) / (Lx*Ly)
    angle = np.arccos(cos_angle)*360/2/np.pi
    return angle

def geometry_nor(data, clear_1):
    left_hand = data['skeleton'][0]['Dxyz'][47][:,0:10000]
    right_hand = data['skeleton'][0]['Dxyz'][19][:,0:10000]
    left_foot = data['skeleton'][0]['Dxyz'][7][:,0:10000]
    right_foot = data['skeleton'][0]['Dxyz'][3][:,0:10000]

    left_shoulder = data['skeleton'][0]['Dxyz'][44][:,0:10000]
    right_shoulder = data['skeleton'][0]['Dxyz'][16][:,0:10000]

    spine = data['skeleton'][0]['Dxyz'][12][:,0:10000]
    hip = data['skeleton'][0]['Dxyz'][0][:,0:10000]
    geometry = np.zeros([3, len(clear_1)])
    p = 0
    for q in clear_1:
        p1 = left_shoulder[:, q].T
        p2 = right_shoulder[:, q].T
        p3 = left_foot[:, q].T
        a,b,c,d = threepoints_flat(p1,p2,p3)
        d_left_hand = (a*left_hand[0,q]+b*left_hand[1,q]+c*left_hand[2,q]+d) / np.sqrt(a**2+b**2+c**2)
        d_right_hand = (a*right_hand[0,q]+b*right_hand[1,q]+c*right_hand[2,q]+d) / np.sqrt(a**2+b**2+c**2)
        d_right_foot = (a*right_foot[0,q]+b*right_foot[1,q]+c*right_foot[2,q]+d) / np.sqrt(a**2+b**2+c**2)
        if abs(d_right_foot) <= 20:
            d_right_foot = 0
        if d_left_hand < 0:
            d_left_hand = abs(d_left_hand)
        if d_right_hand < 0:
            d_right_hand = abs(d_right_hand)
        if d_right_foot == 0 and d_left_hand > 0 and d_right_hand > 0:
            geometry[0,p] = 1
        elif d_right_foot > 0 and d_left_hand > 0 and d_right_hand > 0:
            geometry[0,p] = 2
        elif d_right_foot < 0 and d_left_hand > 0 and d_right_hand > 0:
            geometry[0,p] = 3
        else:
            geometry[0,p] = 4
        d_left_hand_spine = np.sqrt((left_hand[0,q]-spine[0,q])**2+(left_hand[1,q]-spine[1,q])**2+(left_hand[2,q]-spine[2,q])**2)
        d_left_hand_hip = np.sqrt((left_hand[0,q]-hip[0,q])**2+(left_hand[1,q]-hip[1,q])**2+(left_hand[2,q]-hip[2,q])**2)
        if d_left_hand_spine < d_left_hand_hip:
            geometry[1,p] = 1
        if d_left_hand_hip < d_left_hand_spine:
            geometry[1,p] = 2
        d_right_hand_spine = np.sqrt((right_hand[0,q]-spine[0,q])**2+(right_hand[1,q]-spine[1,q])**2+(right_hand[2,q]-spine[2,q])**2)
        d_right_hand_hip = np.sqrt((right_hand[0,q]-hip[0,q])**2+(right_hand[1,q]-hip[1,q])**2+(right_hand[2,q]-hip[2,q])**2)
        if d_right_hand_spine < d_right_hand_hip:
            geometry[2,p] = 1
        if d_right_hand_hip < d_right_hand_spine:
            geometry[2,p] = 2
        p = p + 1
    return geometry

def min_index(min_value, series_score, index, length, last_cut):
    score_index = []
    for a in range(0, len(series_score)):
        if series_score[a] == min_value:
            score_index.append(a)
    dis = []
    for b in score_index:
        dis.append(abs(length-(index[b] - last_cut)))
    dis_index = dis.index(min(dis))
    index_cut = index[score_index[dis_index]]
    return index_cut

def classify(som, data, sample, label):
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

def clear(num, data, start, end):
    if num < 10:
        ss = 'sj_half_0{p}'
    else:
        ss = 'sj_half_{p}'
    data['skeleton'] = data.pop(ss.format(p=num))
    a = data['skeleton'][0]['Dxyz'][19]
    m = 50
    c = copy.copy(a)
    A = pointfilter(m, c)
    action = A[:,start:end:5]
    k = np.zeros(action.shape[1]-2)
    a_thre = 0.1

    for i in range(0,action.shape[1]-2):
        pc, k[i] = points2circle(action[:,i].T, action[:,i+1].T, action[:,i+2].T)

    cut_1 = cut(k, a_thre, start)

    max_cutlen = 50
    min_cutlen = 15
    inter = 30
    clear_1 = merge(cut_1,max_cutlen,inter,min_cutlen,start,end)
    return clear_1, A

def SOM_train(X_train1,size,max_iter):
    N = X_train1.shape[0]
    M = X_train1.shape[1]
    som = MiniSom(size, size, M, sigma=1, learning_rate=0.5)
    som.pca_weights_init(X_train1)
    som.train_batch(X_train1, max_iter, verbose=False)
    winner_coordinates = np.array([som.winner(x) for x in X_train1]).T
    som_shape = (size, size)
    cluster_train = np.ravel_multi_index(winner_coordinates, som_shape)
    return som, cluster_train

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def sample_template(sample, template):
    template_point = []
    for i in sample:
        template_point.append(find_nearest(template, i))
    return template_point


##### CUT FRAMES & GEOMETRY FEATURES
sample = [[250,595,1930,2390,3155,3785,5150,5720,6280,6910,7485],\
        [770,1255,2580,2980,3570,4165,5385,5805,6285,6915,7425],\
        [445,800,2015,2410,3060,3550,4670,5100,5495,6005,6465],\
        [695,1020,2280,2695,3395,4095,5465,5955,6420,7050,7560],\
        [435,900,2425,2890,3745,4495,5785,6330,6875,7495,8140],\
        [455,915,2675,3160,4025,4780,6060,6545,7060,7680,8355],\
        [300,625,1910,2330,2955,3540,4720,5150,5555,6070,6605],\
        [285,925,2680,3295,4220,5010,6450,7005,7645,8275,8965],\
        [420,950,2785,3370,4265,4970,6345,6835,7405,8110,8805]]
gesture_cp = [[1,2,2],[2,1,1],[1,1,1],[1,2,1],[1,1,1],[3,2,1],[1,1,1],[3,1,1],[3,1,1]]
sample_num = [1,3,5,6,7,9,10,11]
##### TRAIN DATA
train_num = 8
dataFile = '/home/xuleiyang/workspace/TaiChi/skeleton/sj_half_0{m}'
data = scio.loadmat(dataFile.format(m=train_num))
start = sample[train_num-1][0]
end = sample[train_num-1][-1]
clear_1, A = clear(sample_num[train_num-1], data, start, end)
feature_number = 11       
X_train1 = train_sample_feature(clear_1, A, feature_number)
size = 3
max_iter = 8000
#### TRAJECTORTY PRIMITIVES CLUSTER
som, cluster_train = SOM_train(X_train1, size, max_iter)
#### TEMPLATE
template_point = sample_template(sample[train_num-1], clear_1)

##### TEST DATA
test_seq_num = [1,2,3,4,5,6,7]
frame_wise_accuracy = np.zeros([len(test_seq_num)])
IoU = np.zeros([len(test_seq_num)])
IoD = np.zeros([len(test_seq_num)])
acc = np.zeros([len(test_seq_num)])
count = 0
for test_num in test_seq_num:
    dataFile1 = '/home/xuleiyang/workspace/TaiChi/skeleton/sj_half_0{n}'
    data1 = scio.loadmat(dataFile1.format(n=test_num))
    start1 = sample[test_num-1][0]
    end1 = sample[test_num-1][-1]
    clear_2, B = clear(sample_num[test_num-1], data1, start1, end1)
    #### TRAJECTORY PRIMITIVE SEQUENCES
    X_train2 = train_sample_feature(clear_2, B, feature_number)
    cluster_test = np.array(classify(som, X_train2, X_train1, cluster_train))

    #### GEOMETRY FEATURES
    geometry = geometry_nor(data1, clear_2)

    #### SEGMENTATION FRAMES
    test_sample_cp_clear = [0]
    test_sample_cp = [start1]
    for ct_num in range(len(template_point)-2):
        gesture = gesture_cp[ct_num]
        template = cluster_train[template_point[ct_num]:template_point[ct_num+1]]
        last_cut = test_sample_cp_clear[-1]
        index_1 = np.where(geometry[0,:] == gesture[0])
        index_2 = np.where(geometry[1,:] == gesture[1])
        index_3 = np.where(geometry[2,:] == gesture[2])
        index = np.intersect1d(np.intersect1d(index_1[0][index_1[0]>last_cut],index_2[0][index_2[0]>last_cut]), index_3[0][index_3[0]>last_cut])
        series_score = []
        for ii in index:
            series =cluster_test[last_cut:ii]
            series_score.append(h_dtw(template,series))
        min_value = min(series_score)
        length = len(template)
        index_cut = min_index(min_value, series_score, index, length, last_cut)
        test_sample_cp_clear.append(index_cut)
        cut_point = clear_2[index_cut]
        test_sample_cp.append(cut_point)
        # print(cut_point)
        pass

    test_sample_cp.append(end1)

    #### METRICS
    frame_wise_accuracy[count] = 1-(np.sum(abs(np.array(sample[test_num-1][1:-1])-np.array(test_sample_cp[1:-1]))) / (sample[test_num-1][-1]-sample[test_num-1][0]))
    IoU_action = []
    IoD_action = []
    acc_action = []
    for p in range(10):
        c = [sample[test_num-1][p], test_sample_cp[p], sample[test_num-1][p+1], test_sample_cp[p+1]]
        c.sort()
        acc_action.append((c[-2]-c[1])/(sample[test_num-1][p+1]-sample[test_num-1][p]))
        IoU_action.append((c[-2]-c[1])/(c[-1]-c[0]))
        IoD_action.append((c[-2]-c[1])/(test_sample_cp[p+1]-test_sample_cp[p]))

    IoU[count] = np.mean(IoU_action)
    IoD[count] = np.mean(IoD_action)
    acc[count] = np.mean(acc_action)
    count = count + 1
