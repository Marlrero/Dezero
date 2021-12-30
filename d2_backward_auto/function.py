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
        output.set_creator(self)  # 변수에 창조자는 이 함수가 됨 (연결을 동적으로 하는 것임)
        
        self.input = input    # 입력변수 저장 (역전파를 위해 순전파 데이터 저장 필요)
        self.output = output  # 출력변수 저장 (앞으로를 위해 출력변수를 따로 저장함)
        
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
        # 순전파와 역전파 동시 구현
        print("forward!")
        A = Square()
        B = Exp()
        C = Square()
        
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        print(a, b, y)
        
        # 계산 그래프가 거꾸로 올라가는지 확인(역전파되는지 확인) -> Define by Run!
        # assert문: 평가 결과가 True가 아니면 예외 발생 코드
        assert y.creator == C   # y의 창조자는 C함수가 맞는가?
        assert y.creator.input == b  # C함수의 입력은 b가 맞는가?
        assert y.creator.input.creator == B  # C함수의 입력의 창조자는 B함수가 맞는가?
        assert y.creator.input.creator.input == a  # B함수의 입력은 a가 맞는가?
        assert y.creator.input.creator.input.creator == A  # B함수의 입력의 창조자는 A함수가 맞는가?
        assert y.creator.input.creator.input.creator.input == x  # A함수의 입력은 x가 맞는가?
        # 예외가 발생하지 않으면 위 문장은 모두 True!        
        
        # 변수와 함수의 관계(창조자)를 이용한 역전파 구현
        # x -> A -> a -> B -> b -> C -> y
        y.grad = np.array(1.0)
        C = y.creator   # 변수 y의 창조자는 C함수이다
        b = C.input     # C함수의 입력은 b이다
        b.grad = C.backward(y.grad)  # b에 대한 C함수의 미분값은 출력 y를 사용함 (dy/dy * dy/db = dy/db)
        print(f"b.grad: {b.grad}")
        
        B = b.creator   # 변수 b의 창조자는 B함수이다
        a = B.input     # B함수의 입력은 a이다
        a.grad = B.backward(b.grad)  # a에 대한 B함수의 미분값은 출력 b를 사용함 (dy/db * db/da = dy/da)
        print(f"a.grad: {a.grad}")
        
        A = a.creator   # 변수 a의 창조자는 A함수이다
        x = A.input     # A함수의 입력은 x이다
        x.grad = A.backward(a.grad)  # x에 대한 A함수의 미분값은 출력 a를 사용함 (dy/da * da/dx = dy/dx)
        print(f"x.grad: {x.grad}")
    
    main()