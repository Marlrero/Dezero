import unittest

import numpy as np

import sys
sys.path.append("..")
from d5_function_upgrade.variable import Variable
from d5_function_upgrade.function import square

class SquareTest(unittest.TestCase):
    """Square 함수 테스트
    
    [단위 테스트 하는 법]
    1. unittest.TestCase를 상속한다.
    2. test로 시작하는 메소드를 만든다.
    3. 함수의 출력이 expected(기댓값)과 같은지 확인
        -> self.assertEqual()
    """
    
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()  # x.grad 갱신
        expected = np.array(6.0)  # 2x = 2 * 3 = 6
        self.assertEqual(x.grad, expected)
        
# 테스트 실행 방법
# python -m unittest ./square_test.py