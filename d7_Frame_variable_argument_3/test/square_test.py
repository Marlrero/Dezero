import unittest
import logging

import numpy as np

import sys
sys.path.append("..")
from core_simple.calculation import add
from core_simple.variable import Variable
from core_simple.function import square

class SquareTest(unittest.TestCase):
    """Square 함수 테스트
    
    [단위 테스트 하는 법]
    1. unittest.TestCase를 상속한다.
    2. test로 시작하는 메소드를 만든다.
    3. 함수의 출력이 expected(기댓값)과 같은지 확인
        -> self.assertEqual()
    """

    def test_backward(self):
        log = logging.getLogger("SquareTest.test_backward()")
        
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        
        z = add(square(x), square(y))  
        z.backward()
        
        log.debug(f"{z.data}, {x.grad}, {y.grad}")
        self.assertEqual(z.data, 13) # 2^2 + 3^2 = 4 + 9 = 13
        self.assertEqual(x.grad, 4)  # 2x = 2 * 2 = 4
        self.assertEqual(y.grad, 6)  # 2y = 2 * 3 = 6

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("SquareTest.test_backward()").setLevel(logging.DEBUG)
    unittest.main()
        
# 테스트 실행 방법
# python ./square_test.py