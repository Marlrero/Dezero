import numpy as np
from variable import Variable
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
        return output

    def forward(self, x: Variable) -> NotImplementedError:
        """순전파 함수(__call__ 함수에서 위임)

        Args:
            x (Variable): 입력

        Returns:
            NotImplementedError: 현재 구현된 바 없음 (상속해서 오버라이딩되기 때문!)
        """
        raise NotImplementedError

class Square(Function):
    """제곱 함수 (Function 클래스 상속)

    Inheritance:
        Function
    """
    def forward(self, x):
        return x ** 2


if __name__ == '__main__':
    def main():
        x = Variable(np.array(10))
        # f = Function()  # __call__ special method invoke!
        f = Square()
        y = f(x)
        
        print(type(y))
        print(y.data)
        
    
    main()