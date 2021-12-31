from math import gamma
from scipy.sparse.linalg.isolve.iterative import cg
from tools.functions import SSE, Huber, Linf, L1, L2, L2S, LogSmoothing
from tools.optimization import AGM, NewtonCG, LBFGS
import numpy as np
from tools.tests import v_measure, solution_to_cluster
import matplotlib.pyplot as plt


class MyCluster:

    def __init__(self, data, model, minimizer, lambda_, tol=10**(-3), max_iter=np.infty, **param):
        self._data = data
        self._data_shape = data.shape
        self._n = self._data_shape[1]
        self._d = self._data_shape[0]
        self._model_name = model
        self._lambda = lambda_
        self._minimizer_name = minimizer
        self._tol = tol

        self._max_iter = max_iter
        self._param = param
        self._init = self._data

        self._function = self._get_function()
        self._minimizer = self._get_minimizer()

    def solution_to_cluster(self, epsilon=0.01):
        self._epsilon = epsilon
        self.cluster = solution_to_cluster(self.solution, self._epsilon)
        self.n_cluster = len(list(set(self.cluster)))

    def get_result_panel(self, label, beta=1):

        print(f"--------\nResult\n--------")

        print(f"+ Model: {self._model_name}")
        print(f"  - {f'lambda:':<20}{self._lambda:>10}")
        if self._model_name == "Huber-type":
            print(f"  - {f'delta:':<20}{self._delta:>10}")
        elif self._model_name in ["Weighted", "WeightedL1", "WeightedLinf"]:
            print(f"  - {f'delta:':<20}{self._delta:>10}")
            print(f"  - {f'k:':<20}{self._k:>10}")
            print(f"  - {f'v:':<20}{self._v:>10}")

        print(f"+ Minimizer: {self._minimizer_name}")
        print(f"  - {f'Max Iteration:':<20}{self._max_iter:>10}")
        print(f"  - {'Tolerance:':<20}{self._tol:>10}")
        if self._minimizer_name == "NewtonCG":
            print(f"  - {f'cg max:':<20}{self._cg_max:>10}")
            print(f"  - {f'gamma:':<20}{self._gamma:>10}")
            print(f"  - {f'sigma:':<20}{self._sigma:>10}")
        elif self._minimizer_name == "LBFGS":
            print(f"  - {f's:':<20}{self._s:>10}")
            print(f"  - {f'm:':<20}{self._m:>10}")
            print(f"  - {f'gamma:':<20}{self._gamma:>10}")
            print(f"  - {f'sigma:':<20}{self._sigma:>10}")
            print(
                f"  - {f'max_line_search_iter:':<20}{self._max_line_search_iter:>9}")
        elif self._minimizer_name == "SGD":
            print(f"  - {f'batch_size:':<20}{self._batch_size:>10}")

        print(f"+ Performance:")
        print(f"  - {'Data Shape:':<20}{f'{self._d} x {self._n}':>10}")
        print(f"  - {'Iteration Times:':<20}{self.iter:>10}")
        print(f"  - {'Duration:':<20}{f'{self.duration:.3f}s':>10}")
        print(f"  - {'Number of Clusters:':<20}{self.n_cluster:>10}")
        print(f"  - {'Number of Classes:':<20}{len(list(set(label))):>10}")
        print(f"  - {'Compression Ratio:':<20}{self.compress_ratio:>10.3f}")
        self._homogeneity, self._completeness, self._v_measure = v_measure(
            self.cluster, label, beta)
        print(f"  - {f'Homogeneity:':<20}{self._homogeneity:>10.3f}")
        print(f"  - {f'Completeness:':<20}{self._completeness:>10.3f}")
        print(
            f"  - {f'V-measure(beta={beta:<0}):':<20}{self._v_measure:>10.3f}")

        if self._minimizer_name in ['NewtonCG', 'LBFGS']:
            plt.plot(self._minimizer._re_path)
            plt.xlabel('Iteration')
            plt.ylabel('Relative Error')
            plt.title('Convergence Path')
            plt.show()
        else:
            plt.plot(self._minimizer._d_path)
            plt.xlabel('Iteration')
            plt.ylabel('Norm of Gradient')
            plt.title('Convergence Path')
            plt.show()

    def optimize(self):
        self._result = self._minimizer.get_result()
        self.solution = self._result['solution']
        self.duration = self._result['duration']
        self.iter = self._result['iter']
        self.path = self._result['path']
        self.compress_ratio = self._mean_dist(
            self.solution)/self._mean_dist(self._data)

    def _mean_dist(self, x):
        d, n = x.shape
        diff_mat = x.reshape(d, 1, n) - x.reshape(d, n, 1)
        norm_mat = (((diff_mat).pow(2)).sum(0)).sqrt()

        return float(norm_mat.mean())

    def plot2d(self, data_generator):
        shapes = ['o', '^', '1', 's', '*',
                  'p', 'h', '+', 'x', 'd']
        for c in list(set(self.cluster)):
            data = data_generator._data[:, self.cluster == c]
            plt.scatter(data[0],
                        data[1], c=data_generator._label[self.cluster ==
                                                         c], marker=f'${c}$', cmap='Paired',
                        vmin=0, vmax=max(data_generator._label), alpha=0.8)
            shadow = self.solution[:, self.cluster == c]
            plt.scatter(shadow[0],
                        shadow[1], c='black', marker=f'${c}$', alpha=0.1)

        plt.show()

    def _get_function(self):
        sse = SSE(self._data)
        if self._model_name == 'Huber-type':
            self._delta = self._param['delta']
            norm_term = Huber(delta=self._delta, data_shape=self._data_shape)

        elif self._model_name == 'Weighted':
            self._delta = self._param['delta']
            self._k = self._param['k']
            self._v = self._param['v']
            norm_term = Huber(
                delta=self._delta, data_shape=self._data_shape, k_weighted=self._k, ny=self._v)

        elif self._model_name == 'Linf':
            norm_term = Linf()
        elif self._model_name == 'WeightedLinf':
            self._delta = self._param['delta']
            self._k = self._param['k']
            self._v = self._param['v']
            norm_term = Linf(k_weighted=self._k, ny=self._v)
        elif self._model_name == 'L1':
            norm_term = L1()
        elif self._model_name == 'WeightedL1':
            self._delta = self._param['delta']
            self._k = self._param['k']
            self._v = self._param['v']
            norm_term = L1(k_weighted=self._k, ny=self._v)
        elif self._model_name == 'L2':
            norm_term = L2()
        elif self._model_name == 'L2S':
            norm_term = L2S()
        elif self._model_name == 'LogSmoothing':
            norm_term = LogSmoothing()

        return sse/2+self._lambda*norm_term

    def _get_minimizer(self):
        if self._minimizer_name == 'AGM':
            return AGM(function=self._function, init=self._init, tol=self._tol, max_iter=self._max_iter, use_sgd=False, batch_size=None)

        elif self._minimizer_name == 'SGD':
            self._batch_size = self._param['batch_size']
            return AGM(function=self._function, init=self._init, tol=self._tol, max_iter=self._max_iter, use_sgd=True, batch_size=self._batch_size)

        elif self._minimizer_name == 'NewtonCG':
            self._cg_max = self._param['cg_max']
            self._gamma = self._param['gamma']
            self._sigma = self._param['sigma']

            return NewtonCG(function=self._function, init=self._init, tol=self._tol, max_iter=self._max_iter,
                            cg_max=self._cg_max, gamma=self._gamma, sigma=self._sigma)

        elif self._minimizer_name == 'LBFGS':
            self._s = self._param['s']
            self._m = self._param['m']
            self._gamma = self._param['gamma']
            self._sigma = self._param['sigma']
            self._max_line_search_iter = self._param['max_line_search_iter']

            return LBFGS(function=self._function, init=self._init, tol=self._tol, max_iter=self._max_iter,
                         s=self._s, m=self._m, gamma=self._gamma, sigma=self._sigma, max_line_search_iter=self._max_line_search_iter)
