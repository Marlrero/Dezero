from .function import Function
from .variable import Variable

from typing import Tuple

class Add(Function):
    """두 인수를 더함 (계산 그래프 덧셈 수행)
    """
    def forward(self, x0: Variable, x1: Variable) -> Variable:
        """덧셈 순전파

        Args:
            x0 (Variable): 이항 연산에서 첫번째 항
            x1 (Variable): 이항 연산에서 두번째 항

        Returns:
            Variable: 출력
        """
        #x0, x1 = xs
        y = x0 + x1
                
        #return (y,)
        return y

def add(x0: Variable, x1: Variable):
    """Add 클래스를 함수로

    Args:
        x0 (Variable): 이항 연산에서 첫번째 항
        x1 (Variable): 이항 연산에서 두번째 항
    """
    return Add()(x0, x1)