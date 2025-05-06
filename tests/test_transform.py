from context import dhframe
from dhframe.transform import Transform
from sympy import symbols

theta, d, alpha, a = symbols("theta d alpha a")

def test_transform_fixed():
    
    T = Transform(1,1,1,1)
    assert T.is_fixed
    assert True if not T.arguments else False

def test_transform_rot_joint():
    T = Transform(theta, 1, 0, 0)
    assert not T.is_fixed
    assert T.arguments[0] == theta and len(T.arguments) == 1

def test_transform_matrix():
    T = Transform(theta, 1, 0, 0)
    out = repr(T.matrix)
    expected = """Matrix([
[cos(theta), -sin(theta),   0,   0],
[sin(theta),  cos(theta),   0,   0],
[       0.0,           0,   1,   1],
[       0.0,         0.0, 0.0, 1.0]])"""
    assert out == expected

def test_transform_callable():
    T = Transform(theta, 1, 0, 0)
    out = repr(T.callable(0))
    expected = """array([[ 1., -0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  1.],
       [ 0.,  0.,  0.,  1.]])"""
    assert out == expected
