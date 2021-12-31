from numpy.core.fromnumeric import shape
from scipy.io import loadmat
from scipy.sparse import coo_matrix
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import torch
import os


def coo_to_tensor(data):
    ''' 转换稀疏矩阵为稀疏张量
    Args:
        data [scipy.sparse.coo_matrix]: 原始稀疏矩阵

    Retuerns:
        data [torch.Tensor]: 输入的稀疏张量形式
    '''

    values = data.data
    indices = np.vstack((data.row, data.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = data.shape
    data = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return data


def load_data(name):
    '''读取数据
    Args:
        name [str]: 数据集名称，选项 {'mnist','segment','vowel','wine','self_generated_{id}'}

    Retuerns:
        data [torch.Tensor (d x n, sparse)]:
        label [torch.Tensor (n x 1)]:
    '''
    path = os.path.join('..', 'data', name)
    if name in ['mnist', 'segment', 'vowel', 'wine']:
        data = loadmat(path+"_data.mat")['A']
        label = loadmat(path+"_label.mat")['b'].ravel()
        # if name == 'mnist':
        #     data = coo_matrix(data)

        data = coo_matrix(data)
        data = coo_to_tensor(data)
        label = torch.Tensor(label)
        label = np.array(label, dtype=int)
    else:
        data = coo_matrix(np.loadtxt(path+"_data.txt"))
        data = coo_to_tensor(data)
        label = np.loadtxt(path+"_label.txt").ravel()
        label = torch.Tensor(label)
        label = np.array(label, dtype=int)

    return data.to_dense(), label


class DataGenerator:
    '''用于生成数据的类
    __init__:
        p[int]: 类个数
        field[list(2 x 2)]: 两个维度的定义域，默认1-10，1-10
        np_list [list < int >]: 每个类点个数, 默认每类20个点
        refernce_point_list [list < list < float >>]: 每个类的中心，默认在定义域上从均匀分布中随机生成
        sd_list [list < float >]: 每个类的方差，默认为每一类离最近点距离/分离度(sep)
        sep[float]: 类间数据分离度
        seed[int]: 随机数种子，默认随机生成

    Attributes:
        seed[int]: 当前随机数种子

        _data[torch.Tensor (2 x n, sparse)]: 
        _label[torch.Tensor (n x 1)]:

    Method:
        get_data(): 获取数据
        plot(): 绘制数据图像
        write(id[str]): 将数据写入文件

        _generate_data(): 生成数据
    '''

    def __init__(self, p, field=[[0, 10], [0, 10]], np_list=None, refernce_point_list=None, sd_list=None, sep=None, seed=None):
        self.seed = np.random.randint(10000) if not seed else seed
        self._data, self._label = self._generate_data(
            p, field, np_list, refernce_point_list, sd_list, sep, seed=self.seed)

    def get_data(self):
        return self._data, self._label

    def plot(self):
        plt.scatter(np.array(self._data[0]),
                    np.array(self._data[1]), c=self._label, cmap='Paired')
        plt.show()

    def write(self, id):
        path = os.path.join('..', 'data', 'self_generated_')+id
        np.savetxt(path+"_data.txt", self._data.numpy())
        np.savetxt(path+"_label.txt", self._label)
        plt.show()

    def _generate_data(self, p, field, np_list, refernce_point_list, sd_list, sep, seed):
        np.random.seed(seed)
        np_list = [20]*p if not np_list else np_list  # 默认每类20个点
        refernce_point_list = np.stack((np.random.uniform(field[0][0], field[0][1], p), np.random.uniform(
            field[1][0], field[1][1], p))).T.tolist() if not refernce_point_list else refernce_point_list

        def dist(p): return norm(p[0]-p[1])
        if not sd_list:
            min_dist_list = np.array([min([dist(pc) for pc in np.array(list(zip(
                [refernce_point_list[i]]*(p-1), refernce_point_list[:i]+refernce_point_list[i+1:])))]) for i in range(p)])
            sd_list = min_dist_list/sep

        centers = [[refernce_point_list[i]]*np_list[i] for i in range(p)]
        centers = np.array([i for j in centers for i in j])

        errors = [(np.random.randn(np_list[i], 2)*sd_list[i]).tolist()
                  for i in range(p)]
        errors = np.array([i for j in errors for i in j])

        coo = coo_matrix((centers+errors).T)

        data = coo_to_tensor(coo)

        label = [[range(p)[i]]*np_list[i] for i in range(p)]
        label = np.array([i for j in label for i in j]
                         ).ravel()

        np.random.seed(None)

        return data.to_dense(), label
