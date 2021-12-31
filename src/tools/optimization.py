import numpy as np
import time
from scipy.sparse.linalg import norm
from scipy.sparse import csc_matrix
from IPython import display
import matplotlib.pyplot as plt
from math import sqrt
import torch


class Minimizer:
    '''minimizer类
    __init__:
        function[function]: 优化目标函数
        init[np.array(nd x 1)]: 初始点
        tol[float]: 终止条件
        max_iter[int]: 最多迭代次数

    Attributes
        _function:
        _init:
        _tol:
        _solution[array(init.shape)]: 优化解，和起始点同shape
        _duration[float]: 计算时间
        _iter[int]: 迭代次数
        _path[list<np.array>]: 优化路径

    Method
        get_result: 获取结果

        _apply: 实施优化
        _line_search: 步长策略

    '''

    def __init__(self, function, init, tol, max_iter=np.inf):
        self._function = function
        self._init = init
        self._tol = tol
        self._max_iter = max_iter

        self._solution = None
        self._duration = None
        self._iter = None
        self._path = None
        self._d_path = None

    def get_result(self):
        return {
            'solution': self._solution,
            'duration': self._duration,
            'iter': self._iter,
            'path': self._path
        }

    def _apply(self):
        print('Applying: ')

        print('Complete!')

    def _line_search(self, x, d):
        return 0


class AGM(Minimizer):

    def __init__(self, function, init, tol, max_iter=np.inf, **param):
        super().__init__(function, init, tol, max_iter)
        self.param = param

    def get_result(self):
        self._apply(self.param)
        return {
            'solution': self._solution,
            'duration': self._duration,
            'iter': self._iter,
            'path': self._path
        }

    def _line_search(self, x, d):
        '''计算x处，梯度为d的最优步长

        '''
        return 0

    def _get_alpha(self, **param):
        '''

        '''
        return 1/self._function._lipschitz

    def _get_t(self, **param):
        '''

        '''
        if len(self._tmp_params['t']) == 0:
            return 1
        else:
            tk_1 = self._tmp_params['t'][-1]
            return (1+sqrt(4*tk_1**2+1))/2

    def _get_beta(self, **param):
        '''

        '''

        if len(self._tmp_params['beta']) == 0:
            return 0
        else:
            tk = self._tmp_params['t'][-1]
            tk_1 = self._tmp_params['t'][-2]
        return (tk_1-1)/tk

    def _update_tmp_params(self):
        '''

        '''
        self._tmp_params['alpha'].append(self._get_alpha())
        self._tmp_params['t'].append(self._get_t())
        self._tmp_params['beta'].append(self._get_beta())

    def _apply(self, param):
        '''优化算法

        '''
        if self.param['use_sgd']:
            print('Applying: SGD')
        else:
            print('Applying: AGM')

        start = time.time()
        self._iter = 0

        self._tmp_params = {'beta': [],
                            't': [],
                            'alpha': []}
        self._update_tmp_params()

        self._path = [self._init, self._init]
        self._d_path = []

        d, n = self._init.shape
        if self.param['use_sgd'] is True:
            batch_size = self.param['batch_size']

        best_obj = torch.inf
        while True:
            self._iter += 1

            alpha, beta = self._tmp_params['alpha'][-1], self._tmp_params['beta'][-1]
            xk = self._path[-1]
            xk_1 = self._path[-2]

            y = (xk + beta * (xk - xk_1))

            if self.param['use_sgd']:
                index = torch.randperm(n)[:batch_size]
                d = self._function.grad(y[:, index], index=index)
                y[:, index] = y[:, index] - alpha*d
                self._path.append(y)
            else:
                d = self._function.grad(y)
                self._path.append(y-alpha*d)

            if len(self._path) > 2:
                self._path.pop(0)

            d_norm = d.norm(2)  # 计算梯度norm
            self._d_path.append(d_norm)

            if self.param['use_sgd']:
                obj = self._function.obj(self._path[-1][:, index], index=index)
            else:
                obj = self._function.obj(self._path[-1])
            print(f'\riter:{self._iter}\td_norm:{d_norm:.4f}\tobj:{obj:.4f}',
                  end='', flush=True)  # 输出进度
            if d_norm < self._tol or self._iter >= self._max_iter:  # 迭代终止判断
                break

            # if abs(obj - best_obj)/abs(max(1, best_obj)) < self._tol or self._iter >= self._max_iter:  # 迭代终止判断
                # break
            elif obj < best_obj:
                best_obj = obj

            self._update_tmp_params()

        self._solution = self._path[-1]
        end = time.time()  # 获取终止时间
        self._duration = end-start  # 获取duration

        print('\nComplete!')


class NewtonCG(Minimizer):
    def __init__(self, function, init, tol, max_iter=np.inf, **param):
        super().__init__(function, init, tol, max_iter)
        self._param = param

    def get_result(self):
        self._apply()
        return {
            'solution': self._solution,
            'duration': self._duration,
            'iter': self._iter,
            'path': self._path
        }

    def _apply(self):
        '''优化算法

        '''

        print('Applying: NewtonCG')
        self._cg_max = self._param['cg_max']
        self._gamma = self._param['gamma']
        self._sigma = self._param['sigma']
        self._d = self._init.shape[0]
        self._n = self._init.shape[1]

        start = time.time()  # 获取起始时间
        self._iter = 0
        self._path = [self._init]
        self._d_path = []
        self._re_path = []
        g_k = self._function.grad(self._init)
        best_fx = torch.inf
        x_k = self._path[-1]
        grad_norm = g_k.norm(2)
        self._cg_tol_k = max(1, grad_norm**0.1)*grad_norm

        while True:
            self._iter += 1
            d_k = self._conjugate_gradient(g_k, x_k)

            alpha = self._amijo(d_k, x_k)
            x_k = x_k+alpha*d_k
            g_k = self._function.grad(x_k)
            grad_norm = g_k.norm(2)
            f_x_k = self._function.obj(x_k)

            re = (best_fx - f_x_k).abs() / max(1, abs(best_fx))
            print(f'\riter:{self._iter}\td_norm:{grad_norm:.4f}\tre:{re:.4f}',
                  end='', flush=True)  # 输出进度
            self._re_path.append(re)

            if self._re_path[-1] <= self._tol:
                break
            if best_fx > f_x_k:
                best_fx = f_x_k
            if grad_norm < self._tol or self._iter >= self._max_iter:  # 迭代终止判断
                break

            self._path.append(x_k)
            self._d_path.append(grad_norm)
            self._cg_tol_k = max(1, grad_norm**0.1)*grad_norm

        self._solution = self._path[-1]
        end = time.time()  # 获取终止时间
        self._duration = end-start  # 获取duration

        print('\nComplete!')

    def _amijo(self, d_k, x_k):
        alpha = 1
        # AR condition
        fxk = self._function.obj(x_k)
        for i in np.arange(50):
            xk1 = x_k + alpha*d_k
            fxk1 = self._function.obj(xk1)
            if fxk1 - fxk > self._gamma*alpha*torch.dot(self._function.grad(x_k).ravel(), d_k.ravel()):
                alpha = self._sigma**(i+1)
            else:
                break
        return alpha

    def _conjugate_gradient(self, d_k, x_k):
        d_k = d_k.ravel()
        vj = 0
        vj1 = 0
        rj = d_k
        pj = -rj
        for j in np.arange(self._cg_max):
            Apj = torch.mv(self._function.hessian(x_k), pj)
            quadratic = torch.dot(pj, Apj).squeeze()
            if quadratic <= 0:
                return -d_k.reshape(self._d, self._n) if j == 0 else vj1.reshape(self._d, self._n)
            sigmaj = rj.norm(2).pow(2)/quadratic   # scalar
            vj1 = vj + sigmaj * pj  # dn*1维
            rj1 = rj+sigmaj*Apj  # d*n维
            beta_j1 = rj1.norm(2).pow(2) / rj.norm(2).pow(2)  # 近似计算
            pj1 = -rj1 + beta_j1 * pj
            if rj1.norm(2) < self._cg_tol_k:
                dk = vj1.reshape(self._d, self._n)
                return dk
            # 更新pj,vj,rj
            pj = pj1
            vj = vj1
            rj = rj1
        return vj1.reshape(self._d, self._n)


class LBFGS(Minimizer):
    '''示例类，继承自父类Minimizer
    __init__:
        **param: 该优化算法的具体参数

    Method:
        _line_search: 步长搜索策略，通用性强可以写入父类
        _apply: 优化算法

    '''

    def __init__(self, function, init, tol, max_iter=np.inf, **param):
        super().__init__(function, init, tol, max_iter)
        self.param = param

    def get_result(self):
        self._apply(self.param)
        return {
            'solution': self._solution,
            'duration': self._duration,
            'iter': self._iter,
            'path': self._path
        }

    def _get_rho(self, k, **param):
        '''

        '''
        if k <= 0:
            return self._tmp_params['s'][0]
        else:
            return 1/(1e-6+self._tmp_params['s'][k].T.dot(self._tmp_params['y'][k]))

    def _get_s(self, k, **param):
        '''

        '''
        if k <= 0:
            return self._path[0].reshape(-1)
        else:
            return (self._path[k+1] - self._path[k]).reshape(-1)

    def _get_y(self, k, **param):
        '''

        '''
        if k <= 0:
            return self._path[0].reshape(-1)
        else:
            return (self._tmp_params['grad'][k+1] - self._tmp_params['grad'][k]).reshape(-1)

    def _update_tmp_params(self, k):
        '''

        '''
        self._tmp_params['grad'][k +
                                 1] = self._function.grad(self._path[k+1]).reshape(-1)
        self._tmp_params['s'][k] = self._get_s(k)
        self._tmp_params['y'][k] = self._get_y(k)
        self._tmp_params['rho'][k] = self._get_rho(k)

    def _get_d(self, k):
        q = self._tmp_params['grad'][k].reshape(-1)
        for i in range(k-1, max(0, k-self._m), -1):
            self._tmp_params['alpha'][i] = self._tmp_params['rho'][i] * \
                self._tmp_params['s'][i].T.dot(q)
            q = q - self._tmp_params['alpha'][i] * self._tmp_params['y'][i]
        if self._tmp_params['s'][k-1].T.dot(self._tmp_params['y'][k-1]) >= 0 or k <= self._m:
            self._tmp_params['H'] = (self._tmp_params['s'][k-1].T.dot(self._tmp_params['s'][k-1]) / (
                1e-6+self._tmp_params['s'][k-1].T.dot(self._tmp_params['y'][k-1]))) * torch.eye(q.shape[0])
        r = torch.mv(self._tmp_params['H'], q)
        for i in range(k-1, max(0, k-self._m), -1):
            beta = self._tmp_params['rho'][i] * \
                self._tmp_params['y'][i].T.dot(r)
            r = r + (self._tmp_params['alpha'][i] -
                     beta) * self._tmp_params['s'][i]
        return r

    def _backtracking(self, x, d):
        alpha = self._s
        fx = self._function.obj(x)
        _iter = 0
        while True:
            if self._function.obj(x+alpha*d.reshape(x.shape)) - fx <= self._gamma * alpha * -torch.dot(d.T, d) or _iter > self._line_search_iter:
                return alpha
            else:
                alpha *= self._sigma
            _iter += 1

    def _apply(self, param):
        '''优化算法

        '''
        print('Applying: LBFGS')

        start = time.time()
        self._iter = 0
        self._m = param['m']
        self._s = param['s']
        self._gamma = param['gamma']
        self._sigma = param['sigma']
        self._line_search_iter = param['max_line_search_iter']

        self._tmp_params = {'alpha': {},
                            'rho': {},
                            's': {},
                            'y': {},
                            'grad': {},
                            'H': 0,
                            'd': {}}

        self._path = [self._init]
        self._d_path = [self._function.grad(self._init).norm(2)]
        self._re_path = []
        best_obj = torch.inf
        while True:
            self._iter += 1
            k = self._iter - 1

            xk = self._path[k]
            if self._iter <= self._m:
                dk = - self._function.grad(self._path[k]).reshape(-1)
                self._tmp_params['grad'][k] = dk
                self._tmp_params['d'][k] = dk
                alpha = self._backtracking(xk, dk)
                self._path.append(xk + alpha * dk.reshape(xk.shape))
                self._tmp_params['grad'][k+1] = dk
            else:
                dk = - self._get_d(k)
                alpha = self._backtracking(xk, dk)
                self._path.append(xk + alpha * dk.reshape(xk.shape))

            self._update_tmp_params(k)

            d_norm = self._tmp_params['grad'][k+1].norm(2)  # 计算梯度norm
            self._d_path.append(d_norm)
            obj = self._function.obj(self._path[k+1])
            re = abs(obj - best_obj)/abs(max(1, best_obj))
            print(f'\riter:{self._iter}\td_norm:{d_norm:.4f}\tobj:{obj:.4f}\tre:{re:.4f}',
                  end='', flush=True)  # 输出进度

            self._re_path.append(re)
            if self._re_path[-1] < self._tol or self._iter >= self._max_iter:  # 迭代终止判断
                break
            elif obj < best_obj:
                best_obj = obj
#             if d_norm < self._tol or self._iter >= self._max_iter:  # 迭代终止判断
#                 break
        self._solution = self._path[-1]
        end = time.time()  # 获取终止时间
        self._duration = end-start  # 获取duration

        print('\nComplete!')
