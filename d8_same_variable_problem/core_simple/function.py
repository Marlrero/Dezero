import numpy as np
from .variable import Variable
# https://www.python.org/dev/peps/pep-0008
# https://www.python.org/dev/peps/pep-0007/

def as_array(x) -> np.ndarray:
    """Variable 클래스의 data가 np.ndarray 타입을 강제하는 함수
      (main.py 68번째 라인 참고)

    Args:
        x ([type]): 스칼라(np.int, np.float00)나 리스트 모두 올 수 있음

    Returns:
        [np.ndarray]
    """
    if np.isscalar(x):  # 입력 x가 스칼라면 (0차원이면)
        return np.array(x)  # ndarray로 감싸라
    return x

class Function:
    """함수 클래스
    """
    
    def __call__(self, *inputs: Variable) -> list:
        """함수
            Functional
        Args:
            input (Variable): 입력배열 (가변 인수로 변경)

        Returns:
            list: 결과 (list로 변경)
        """    
        xs = [x.data for x in inputs] # 데이터 꺼내기 -> (가변 인수로 변경)
        #ys = self.forward(xs)   
        # 구체적 계산은 forward(순전파) 함수에 위임
        ys = self.forward(*xs)  # unpacking
        if not isinstance(ys, tuple):
            ys = (ys,)  # tuple이 아닌 경우 tuple로 감싸라!
        outputs = [Variable(as_array(y)) for y in ys] # 결과 변수가 항상 ndarray로 맞춰줌 -> list로 변경
        
        for output in outputs:
            output.set_creator(self)  # 변수에 창조자는 이 함수가 됨 (연결을 동적으로 하는 것임)
        
        self.inputs = inputs    # 입력변수 저장 (역전파를 위해 순전파 데이터 저장 필요)
        self.outputs = outputs  # 출력변수 저장 (앞으로를 위해 출력변수를 따로 저장함)
        
        return outputs if len(outputs) > 1 else outputs[0]  # 리스트 원소가 1개라면 첫 번째 원소 반환

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
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            gy (Variable): [description]

        Returns:
            np.ndarray: [description]
        """
        #x = self.input.data  # Variable.data
        x = self.inputs[0].data
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

# 자세한 방법은 main.py 참고
def square(x):
    return Square()(x)  # 한줄로 작성 가능
def exp(x):
    return Exp()(x)     # 한줄로 작성 가능 