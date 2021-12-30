import numpy as np
# https://www.python.org/dev/peps/pep-0008

# Circular reference problem
# https://brunch.co.kr/@mathpresso/18
# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
from typing import TYPE_CHECKING   

if TYPE_CHECKING:  # 단순히 타입체킹을 위해 필요한 모듈이라면!
    from function import Function

class Variable:  
    """변수
    """
    def __init__(self, data: np.ndarray) -> None:
        """변수

        Args:
            data (np.ndarray): Numpy 다차원배열
        """
        self.data = data
        self.grad = None # Numpy 다차원배열 (역전파 미분값)
        self.creator = None  # 변수에게 있어 함수는 창조자(creator) -> Define by Run
    
    """창조자를 설정하는 함수
    """
    def set_creator(self, func: 'Function'):  # 타입체킹을 위한 Circular reference problem이 발생한다면 문자열로 처리해야 함
        self.creator = func
        
    
    def backward(self):
        f = self.creator  # 변수의 창조자(함수)를 가져온다
        if f is not None: # 창조자가 재귀를 거치다가 None을 만나면(사용자가 Variable 인스턴스 바깥에서 직접 생성한 것임) 멈춘다
            x = f.input   # 창조자에 대해 입력을 가져온다
            x.grad = f.backward(self.grad)  # 창조자에 대해 backward를 호출해 미분값을 계산한다
            x.backward()  # 거슬러 올라가기 위해 그 이전에 변수에 대한 backward를 호출한다 (재귀)
        
    # 내가 만든거 추가
    # Ref: https://shoark7.github.io/programming/python/difference-between-__repr__-vs-__str__
    def __str__(self) -> str:
        """객체를 문자열로 표현하는 함수(내장)

        Returns:
            data (np.ndarray): data 객체 리턴
        """
        return str(self.data)