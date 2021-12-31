class Function:
    '''函数基类，包含函数的grad、hessian公式，支持相加、乘除常数
    Attributes:
        _obj[function]:
        _grad[function]:
        _hessian[function]:

    Method:
        obj: 
        grad: 
        hessian:
        __add__: 重载加号
        __rmul__: 重载左乘常数
        __truediv__: 重载除以常数
    '''

    def __init__(self, obj=None, grad=None, hessian=None):
        self._obj = obj
        self._grad = grad
        self._hessian = hessian

    def obj(self, x, **param):
        if self._obj:
            return self._obj(x, **param)
        else:
            print('This function does not have a objective function.')
            return None

    def grad(self, x, **param):
        if self._grad:
            return self._grad(x, **param)
        else:
            print('This function does not have a gradient.')
            return None

    def hessian(self, x, **param):
        if self._hessian:
            return self._hessian(x, **param)
        else:
            print('This function does not have a Hessian.')
            return None

    def __add__(self, another_function):
        f = Function()
        obj1 = self._obj
        obj2 = another_function._obj
        if obj1 and obj2:
            f._obj = lambda x, **param: obj1(x, **param)+obj2(x, **param)

        grad1 = self._grad
        grad2 = another_function._grad
        if grad1 and grad2:
            f._grad = lambda x, **param: grad1(x, **param)+grad2(x, **param)

        hessian1 = self._hessian
        hessian2 = another_function._hessian
        if hessian1 and hessian2:
            f._hessian = lambda x, **param: hessian1(x, **param)+hessian2(x, **param)

        lipschitz1 = self._lipschitz
        lipschitz2 = another_function._lipschitz
        if lipschitz1 and lipschitz2:
            f._lipschitz = lipschitz1 + lipschitz2

        return f

    def __rmul__(self, c):
        f = Function()
        if self._obj:
            f._obj = lambda x, **param: self._obj(x, **param)*c
        if self._grad:
            f._grad = lambda x, **param: self._grad(x, **param)*c
        if self._hessian:
            f._hessian = lambda x, **param: self._hessian(x, **param)*c
        if self._lipschitz:
            f._lipschitz = self._lipschitz*c

        return f

    def __truediv__(self, c):
        assert c != 0

        f = Function()
        if self._obj:
            f._obj = lambda x, **param: self._obj(x, **param)/c
        if self._grad:
            f._grad = lambda x, **param: self._grad(x, **param)/c
        if self._hessian:
            f._hessian = lambda x, **param: self._hessian(x, **param)/c
        if self._lipschitz:
            f._lipschitz = self._lipschitz/c
        return f
