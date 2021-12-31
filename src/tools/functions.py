import time
from tools.function_basic import Function
# from scipy.sparse.linalg import norm
# from scipy.sparse import csc_matrix
import numpy as np
import torch
# from itertools import combinations


class SSE(Function):
    '''SSE函数

    __init__:
        alpha[np.array (dn x 1)]


    '''

    def __init__(self, alpha):

        def obj(x, **param):
            if 'index' in param.keys():
                index = param['index']
                return (x-alpha[:, index]).norm(2).pow(2)
            else:
                return (x-alpha).norm(2).pow(2)

        def grad(x, **param):
            if 'index' in param.keys():
                index = param['index']
                return 2*(x-alpha[:, index])
            else:
                return 2*(x-alpha)

        self._hessian = lambda x: 2*torch.eye(x.shape[0]*x.shape[1])

        self._obj = obj
        self._grad = grad

        # self._obj = lambda x: (x-alpha).norm(2).pow(2)
        # self._grad = lambda x: 2*(x-alpha)
        # self._hessian = lambda x: 2*torch.eye(x.shape[0]*x.shape[1])
        self._lipschitz = 1


class Huber(Function):
    '''


    '''

    def __init__(self, delta, data_shape, k_weighted=None, ny=0.5):

        def obj(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, 1, n) - x.reshape(d, n, 1)
            norm_mat = diff_mat.pow(2).sum(0).sqrt()
            mask_l2 = norm_mat <= self._delta
            mask_l1 = norm_mat > delta
            if k_weighted is not None:
                index = norm_mat.topk(n-k_weighted)[1]
                weight_mat = (-ny*norm_mat.pow(2)).exp()
                weight_mat = weight_mat.scatter(1, index, 0)
                return (mask_l2 * norm_mat.pow(2) * weight_mat / (2*delta) + mask_l1 * (norm_mat*weight_mat - delta/2)).sum()/2
            else:
                return (mask_l2 * norm_mat.pow(2) / (2*delta) + mask_l1 * (norm_mat - delta/2)).sum()/2
            

        def grad(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, 1, n) - x.reshape(d, n, 1)
            norm_mat = diff_mat.pow(2).sum(0).sqrt()
            mask_l2 = norm_mat <= delta
            mask_l1 = norm_mat > delta
            if k_weighted is not None:
                index = norm_mat.topk(n-k_weighted)[1]
                weight_mat = (-ny*norm_mat.pow(2)).exp()
                weight_mat = weight_mat.scatter(1, index, 0)
                return ((mask_l2 * diff_mat / delta + mask_l1 * diff_mat / norm_mat).nan_to_num(posinf=0) * weight_mat).sum(1)/2
            else:
                return (mask_l2 * diff_mat / delta + mask_l1 * diff_mat / norm_mat).nan_to_num(posinf=0).sum(1)/2

        def hess(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, 1, n) - x.reshape(d, n, 1)
            norm_mat = diff_mat.norm(2, 0)
            mask_l2 = (norm_mat <= delta)
            mask_l1 = (norm_mat > delta)
            eye = torch.eye(d*n, d*n).reshape(d, n, d, n).permute(0, 2, 1, 3)
            hess_l1 = -(norm_mat.pow(-1).reshape(1, 1, n, n).repeat(d, d, 1, 1) - diff_mat.unsqueeze(
                0)*diff_mat.unsqueeze(1) / (norm_mat.pow(3).reshape(1, 1, n, n))).nan_to_num(posinf=0)
            hess_l2 = - \
                ((torch.eye(d, d).repeat(n, n, 1, 1).permute(2, 3, 0, 1)) - eye) / delta

            hess = mask_l1 * hess_l1 + mask_l2 * hess_l2

            hess += -hess.sum(-1).unsqueeze(-1).repeat(1, 1, 1, n) * \
                torch.eye(n, n).reshape(1, 1, n, n).repeat(d, d, 1, 1)
            # hess -= eye / delta

            if k_weighted is not None:
                index = norm_mat.topk(n-k_weighted)[1]
                weight_mat = (-ny*norm_mat.pow(2)).exp()
                weight_mat = weight_mat.scatter(1, index, 0)
                (hess * weight_mat).permute(2, 0, 3, 1).reshape(n*d, n*d)
            
            return hess.permute(2, 0, 3, 1).reshape(n*d, n*d)

        self._delta = delta
        self._obj = obj
        self._grad = grad
        self._hessian = hess
        self._lipschitz = data_shape[1]/delta
        
        
class Linf(Function):
    '''


    '''

    def __init__(self, k_weighted=None, ny=0.5):

        def obj(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, n, 1) - x.reshape(d, 1, n)
            if k_weighted is not None:
                norm_mat = diff_mat.norm(2, 0)
                index = norm_mat.topk(n-k_weighted)[1]
                weight_mat = (-ny*norm_mat.pow(2)).exp()
                weight_mat = weight_mat.scatter(1, index, 0)
                return ((diff_mat).norm(torch.inf, 0) * weight_mat).sum()/2
            else:
                return diff_mat.norm(torch.inf, 0).sum()/2

        def grad(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, n, 1) - x.reshape(d, 1, n)
            max_index = diff_mat.abs().max(0)[1]
            min_index = -max_index+1
            if k_weighted is not None:
                norm_mat = diff_mat.norm(2, 0)
                index = norm_mat.topk(n-k_weighted)[1]
                weight_mat = (-ny*norm_mat.pow(2)).exp()
                weight_mat = weight_mat.scatter(1, index, 0)
                return (torch.stack((diff_mat[0] * min_index, diff_mat[1] * max_index)).sign() * weight_mat).sum(-1)
            else:
                return torch.stack((diff_mat[0] * min_index, diff_mat[1] * max_index)).sign().sum(-1)

        self._obj = obj
        self._grad = grad
        self._hessian = None
        self._lipschitz = 1    
        
class L1(Function):
    '''


    '''

    def __init__(self, k_weighted=None, ny=0.5):

        def obj(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, n, 1) - x.reshape(d, 1, n)
            if k_weighted is not None:
                norm_mat = diff_mat.norm(2, 0)
                index = norm_mat.topk(n-k_weighted)[1]
                weight_mat = (-ny*norm_mat.pow(2)).exp()
                weight_mat = weight_mat.scatter(1, index, 0)
                return (diff_mat.norm(1, 0) * weight_mat).sum()/2
            else:
                return (diff_mat.norm(1, 0)).sum()/2

        def grad(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, n, 1) - x.reshape(d, 1, n)
            if k_weighted is not None:
                norm_mat = diff_mat.norm(2, 0)
                index = norm_mat.topk(n-k_weighted)[1]
                weight_mat = (-ny*norm_mat.pow(2)).exp()
                weight_mat = weight_mat.scatter(1, index, 0)
                return (diff_mat.sign() * weight_mat).sum(-1)
            else:
                return diff_mat.sign().sum(-1)

        self._obj = obj
        self._grad = grad
        self._hessian = None
        self._lipschitz = 1

        
class L2S(Function):
    '''


    '''

    def __init__(self):

        def obj(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, n, 1) - x.reshape(d, 1, n)
            norm_mat = diff_mat.pow(2).sum(0).sqrt()
            return norm_mat.pow(2).sum()/2
            

        def grad(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, n, 1) - x.reshape(d, 1, n)
            norm_mat = diff_mat.pow(2).sum(0).sqrt()
            return diff_mat.sum(1)/2

        self._obj = obj
        self._grad = grad
        self._hessian = None
        self._lipschitz = 1


class L2(Function):
    '''


    '''

    def __init__(self):

        def obj(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, 1, n) - x.reshape(d, n, 1)
            norm_mat = diff_mat.pow(2).sum(0).sqrt()
            return (norm_mat).sum()/2
            

        def grad(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, 1, n) - x.reshape(d, n, 1)
            norm_mat = diff_mat.pow(2).sum(0).sqrt()
            return (diff_mat / norm_mat).nan_to_num(posinf=0).sum(1)/2

        self._obj = obj
        self._grad = grad
        self._hessian = None
        self._lipschitz = 1


class LogSmoothing(Function):
    '''


    '''

    def __init__(self):

        def obj(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, 1, n) - x.reshape(d, n, 1)
            norm_mat = diff_mat.pow(2).sum(0).sqrt()
            return torch.log(1+norm_mat.pow(2)).sum()/2
            

        def grad(x, **param):
            d, n = x.shape
            diff_mat = x.reshape(d, 1, n) - x.reshape(d, n, 1)
            norm_mat = diff_mat.pow(2).sum(0).sqrt()
            return (2 * diff_mat / (1+norm_mat.pow(2))).sum(-1)

        self._obj = obj
        self._grad = grad
        self._hessian = None
        self._lipschitz = 1