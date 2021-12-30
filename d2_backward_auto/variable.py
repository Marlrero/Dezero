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
        
    # 내가 만든거 추가
    # Ref: https://shoark7.github.io/programming/python/difference-between-__repr__-vs-__str__
    def __str__(self) -> str:
        """객체를 문자열로 표현하는 함수(내장)

        Returns:
            data (np.ndarray): data 객체 리턴
        """
        return str(self.data)

if __name__ == '__main__':
    def main():
        data = np.array(1.0)
        x = Variable(data)
        print(x.data)
        
        x.data = np.array(2.0)
        print(x.data)
    
    main()