import numpy as np
from .variable import Variable
# https://www.python.org/dev/peps/pep-0008
# https://www.python.org/dev/peps/pep-0007/

class Function:
    """함수 클래스
    """
    
    '''
    def __call__(self, input: Variable) -> Variable:
        """함수
            Functional
        Args:
            input (Variable): 입력배열

        Returns:
            Variable: 결과
        """
        x = input.data        # 데이터 꺼내기
        y = x**2              # 계산
        output = Variable(y)  # 결과 변수에 넣기
        return output
    '''
    
    def __call__(self, input: Variable) -> Variable:
        """함수
            Functional
        Args:
            input (Variable): 입력배열

        Returns:
            Variable: 결과
        """
        x = input.data        # 데이터 꺼내기
        y = self.forward(x)   # 구체적 계산은 forward(순전파) 함수에 위임
        output = Variable(y)  # 결과 변수에 넣기
        
        self.input = input    # 입력변수 저장 (역전파를 위해 순전파 데이터 저장 필요)
        return output

    def forward(self, x: Variable):
        """순전파 함수(__call__ 함수에서 위임)

        Args:
            x (Variable): 입력

        Raises:
            NotImplementedError: 현재 구현된 바 없음 (상속해서 오버라이딩되기 때문!)
        """
        raise NotImplementedError
    
    def backward(self, gy):
        """역전파 함수(상속받은 클래스에 위임)

        Args:
            gy ([type]): 미분값

        Raises:
            NotImplementedError: 현재 구현된 바 없음 (상속해서 오버라이딩되기 때문!)
        """
        raise NotImplementedError

class Square(Function):
    """제곱함수 (Function 클래스 상속)

    Inheritance:
        Function
    """
    def forward(self, x: Variable) -> np.float64:
        """제곱함수 순전파

        Args:
            x (Variable): 입력

        Returns:
            np.float64: Numpy square
        """
        return x ** 2
    
    #TODO backward 구현
    def backward(self, gy: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            gy (Variable): [description]

        Returns:
            np.ndarray: [description]
        """
        x = self.input.data  # Variable.data
        gx = 2 * x * gy      # 미분!
        return gx        
    
class Exp(Function):
    """제곱 함수 (Function 클래스 상속)

    Inheritance:
        Function
    """
    def forward(self, x: Variable) -> np.float64:
        """지수함수 순전파

        Args:
            x (Variable): 입력

        Returns:
            np.float64: Numpy exponential function
        """
        return np.exp(x)
    
    #TODO backward 구현
    def backward(self, gy: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            gy (Variable): [description]

        Returns:
            np.ndarray: [description]
        """
        x = self.input.data   # Variable.data
        gx = np.exp(x) * gy   # 미분!
        return gx        


if __name__ == '__main__':
    def main():
        # 제곱함수 확인
        x = Variable(np.array(10))
        # f = Function()  # __call__ special method invoke!
        f = Square()
        y = f(x)
        
        print(type(y))
        print(y.data)
        
        
        # 지수함수 확인
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
        
        # 순전파 구현
        print("forward!")
        A = Square()
        B = Exp()
        C = Square()
        
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        c = C(b)
        print(a, b, c)
        
        # 역전파 구현
        print("backward")
        y.grad = np.array(1.0)
        b.grad = C.backward(y.grad)
        a.grad = B.backward(b.grad)
        x.grad = A.backward(a.grad)
        print(y.grad, b.grad, a.grad, x.grad)
        
    
    main()