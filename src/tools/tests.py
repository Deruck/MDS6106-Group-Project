from itertools import product
from collections import deque
import numpy as np
from numpy.linalg import norm


def solution_to_cluster(x, epsilon):

    x = x.numpy()
    n = x.shape[1]
    index = np.array(range(n))

    x_comb = np.array(list(product(x.T, x.T)))
    def sub(x): return x[:, 0, :]-x[:, 1, :]
    x_sub = sub(x_comb)
    x_norm = np.apply_along_axis(norm, 1, x_sub)

    dist_matrix = (x_norm.reshape(n, n) <= epsilon)
    cluster_result = np.array([0]*n)

    d = deque()
    cluster_id = 1
    for ind in index:
        if not cluster_result[ind]:
            cluster_member = set()
            d.append(ind)
            while d:
                p_index = d.popleft()
                cluster_member.add(p_index)
                new_index = index[dist_matrix[p_index]]
                for i in new_index:
                    if not i in cluster_member:
                        d.append(i)
            cluster_result[np.in1d(index, list(cluster_member))] = cluster_id
            cluster_id += 1

    return cluster_result


def v_measure(cluster, label, beta):
    h = _homogeneity(cluster, label)
    c = _completeness(cluster, label)

    return h,c,((1+beta)*h*c)/(beta*h+c)


"""
----------------------------------------------------------------------------------------------------------------
"""


def _entropy(X):
    classes = list(set(X))
    N = X.shape[0]
    A = np.array([sum(X == c) for c in classes])
    plog = np.vectorize(lambda x: -(x/N)*np.log(x/N) if x else 0)
    return np.apply_along_axis(plog, 0, A).sum()


def _joint_entropy(X, Y):
    N = X.shape[0]
    X_Y = np.stack([X, Y])
    X_classes = list(set(X))
    Y_classes = list(set(Y))

    A = np.array([[sum(np.apply_along_axis(np.sum, 1, X_Y.T == np.array(
        [Xc, Yc])) == 2) for Yc in Y_classes] for Xc in X_classes]).ravel()

    plog = np.vectorize(lambda x: -(x/N)*np.log(x/N) if x else 0)
    return np.apply_along_axis(plog, 0, A).sum()


def _conditional_entropy(X, Y):
    return _joint_entropy(X, Y)-_entropy(Y)


def _homogeneity(cluster, label):
    if _entropy(label) == 0:
        return 1
    else:
        return 1-_conditional_entropy(label, cluster)/_entropy(label)


def _completeness(cluster, label):
    if _entropy(cluster) == 0:
        return 1
    else:
        return 1-(_conditional_entropy(cluster, label)/_entropy(cluster))
