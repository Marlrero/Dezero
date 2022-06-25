import numpy as np

import sys
sys.path.append('..')
from d1_primary.numerical_diff import numerical_diff
from d5_function_upgrade.function import square
from d5_function_upgrade.variable import Variable

from square_test import SquareTest

class SquareTestAdv(SquareTest):
    """SquareTest에서 기울기 확인 자동테스트
    """
    
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        #x = Variable(np.array(3.0))
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()  # x.grad 갱신
        #expected = np.array(6.0)  # 2x = 2 * 3 = 6
        
        expected = numerical_diff(square, x)
        #self.assertEqual(x.grad, expected)
        
        # np.allclose -> |a - b| <= (atol + rtol * |b|)
        # atol = 1e-08, rtol = 1e-05
        # refer: https://cs231n.github.io/neural-networks-3/
        # 
        self.assertTrue(np.allclose(x.grad, expected))