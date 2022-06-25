import numpy as np

from .function import Exp, Function, Square
from .variable import Variable

# 중심차분(오차 발생)
def numerical_diff(f: Function, 
                   x: Variable, 
                   eps: float = 1e-4) -> np.float64:
    """중심차분 함수(미분)
    수치미분은 계산량이 많아 사용 불가! -> 역전파 사용
    역전파 확인 시 주로 사용
    
    Args:
        f (Function): 함수
        x (Variable): 변수
        eps (float, optional): 아주 작은 값. 엡실론. Defaults to 1e-4.

    Returns:
        result (np.float64): 중심차분 결과
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    #print(type((y1.data - y0.data) / (2 * eps)))
    return (y1.data - y0.data) / (2 * eps)

# 합성함수 미분(chain rule)
def com_func(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x))) # 합성함수

if __name__ == '__main__':
    def main():
        f = Square()
        x = Variable(np.array(2.0))
        dy = numerical_diff(f, x) # f'(x) = 2x = 2 * 2.0 = 4.0
        print(dy)
        
        x = Variable(np.array(0.5))
        dy = numerical_diff(com_func, x)
        print(dy)
        
    main()
    