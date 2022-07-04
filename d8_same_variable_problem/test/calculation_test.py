import unittest

import numpy as np

import sys
sys.path.append("..")
from core_simple.calculation import Add, add
from core_simple.variable import Variable

class AddTest(unittest.TestCase):
    """Add 클래스 테스트"""
    '''
    def test_forward(self):
        xs = [Variable(np.array(2)), Variable(np.array(3))]
        f = Add()
        ys = f(xs)  # tuple
        y = ys[0]
        
        expected = 5
        self.assertEqual(y.data, expected)
    '''
    '''
    def test_forward_variable_arguments(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        f = Add()
        y = f(x0, x1)
        
        expected = 5
        self.assertEqual(y.data, expected)
    '''
    def test_forward_unpacking_and_impl_func(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        
        expected = 5
        self.assertEqual(y.data, expected)
        
# 테스트 실행 방법
# python -m unittest calculation_test