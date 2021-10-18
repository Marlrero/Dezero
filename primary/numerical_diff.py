import numpy as np

from variable import Variable
from function import Function, Square

# 중심차분(오차 발생)
def numerical_diff(f: Function, x: Variable, eps: float = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

if __name__ == '__main__':
    def main():
        f = Square()
        x = Variable(np.array(2.0))
        dy = numerical_diff(f, x) # f'(x) = 2x = 2 * 2.0 = 4.0
        print(dy)
        
    main()
    