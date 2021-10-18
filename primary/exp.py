import numpy as np

from variable import Variable
from function import Function, Square

class Exp(Function):
    def forward(self, x: Variable) -> np.exp:
        """지수함수

        Args:
            x (Variable): 입력

        Returns:
            np.exp: Numpy exponential function
        """
        return np.exp(x)
    

if __name__ == '__main__':
    def main():
        x = Variable(np.array([2]))
        f = Exp()
        y = f(x) 
        #print(y.data)
        
        # Variable의 __str__ 구현 후
        print(y)
        
        # 합성 함수 만들기!
        A = Square()
        B = Exp()
        C = Square()
        
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        print(y)
        
    main()