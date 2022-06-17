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

def pointfilter(a):
    length = a.shape[1]
    for m in range(2,(length-2)):
        a[:,m] = (a[:,m-2]+a[:,m-1]+a[:,m]+a[:,m+1]+a[:,m+2])/5
    B = a[:,2:length-2]
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

def closest(mylist, number):
    answer = []
    for i in mylist:
        answer.append(abs(number-i))
    return answer.index(min(answer))

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



for a in range(1,11):
    for c in range(0,20):
        data_save = {}
        dataFile = '/home/xuleiyang/workspace/TaiChi/action/action.npy'
        s = 'a{m}'
        a1 = np.load(dataFile, allow_pickle=True).item()
        aa = a1[s.format(m=a)][c][19]
        action = pointfilter(aa)
        for epl in np.arange(0.1, 10, 0.1):
            points = rdp(action, epl)
            index1 = []
            for i in range (len(points)):
                index1.append(np.where(action[0,:] == points[i][0])[0][0])
            length = len(index1)
            m = 'epl_{a}'
            mm = m.format(a=format(epl, '.2f'))
            data_save[mm] = index1
            pass
        n = '/home/xuleiyang/workspace/TaiChi/action/rdp_data/a{a}_t{c}.npy'
        nn = n.format(a=a,c=c)
        np.save(nn, data_save)
        pass
