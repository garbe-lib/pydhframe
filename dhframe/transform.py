import numpy as np
from sympy import lambdify, Matrix, Symbol, cos, sin
from typing import Callable

class Transform:
    
    def __init__(self, theta, d, alpha, a):
        """
        Construct transform from frame n-1 to frame n, using
        Denavit-Hartenberg parameters.
        """
    
        ct = cos(theta).simplify()
        ca = cos(alpha).simplify()
        st = sin(theta).simplify()
        sa = sin(alpha).simplify()
        self._matrix = Matrix([[ct, -st*ca, st*sa, a*ct],
                               [st, ct*ca, -ct*sa, a*st],
                               [0., sa, ca, d],
                               [0., 0., 0., 1.]])
        
        self._parameters = P = [theta, d, alpha, a]

        self._parametric = [not isinstance(x, Symbol) for x in P]
        
        self._arguments = [x for x in P if isinstance(x, Symbol)]

    @property
    def arguments(self) -> list:
        """
        Returns list of variable arguments used in transform definition.
        """
        
        return self._arguments

    @property
    def callable(self) -> Callable:
        """
        Return function generating transform matrix if arguments are present.
        """
        if self.is_fixed:
            return lambda: self._matrix

        else:
            return lambdify(self._arguments, self._matrix)

    @property
    def is_fixed(self) -> bool:
        """
        Returns True, if no parameters are variable arguments. Otherwise, False.
        """

        return not self._matrix.is_symbolic()

    @property
    def matrix(self) -> Matrix:
        """
        Transformation matrix from frame n-1 to frame n.
        """
        
        return self._matrix

    @property
    def parameters(self) -> list:
        """
        List of the parameters theta, d, alpha and a.
        """
        
        return self._parameters

    @property
    def parametric(self) -> list[bool]:
        """
        Defines which parameters are not arguments.
        """
        
        return self._parametric

    @property
    def inverse(self) -> Matrix:
        """
        Transformation matrix from frame n to frame n-1.
        """
        
        theta, d, alpha, a = self._arguments

        ct = cos(theta).simplify()
        ca = cos(alpha).simplify()
        st = sin(theta).simplify()
        sa = sin(alpha).simplify()

        return Matrix([[ct, st, 0, -a],
                       [-st*ca, ct*ca, sa, -d*sa],
                       [sa*st, -ct*sa, ca, -d*ca],
                       [0, 0, 0, 1]])

    def __call__(self, *args):
        """
        Applies arguments to transformation and returns fixed matrix version.
        """
        if self.is_fixed:
            return np.array(self.matrix, dtype= np.float64)

        else:
            return np.array(self.callable(*args), dtype= np.float64)

class RotZ(Transform):
    """
    Joint angle.
    """

    def __init__(self, theta):
        super().__init__(theta, 0, 0, 0)

class RotX(Transform):
    """
    Twist along link axis.
    """
    
    def __init__(self, alpha):
        super().__init__(0, 0, alpha, 0)

class TransZ(Transform):
    """
    Joint distance.
    """
    
    def __init__(self, d):
        super().__init__(0, d, 0, 0)

class TransX(Transform):
    """
    Link length.
    """
    
    def __init__(self, a):
        super().__init__(0, 0, 0, a)

