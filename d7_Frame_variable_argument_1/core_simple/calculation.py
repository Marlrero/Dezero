from .function import Function
from .variable import Variable

from typing import Tuple

class Add(Function):
    """두 인수를 더함 (계산 그래프 덧셈 수행)
    """
    def forward(self, xs: Tuple[Variable, ...]) -> Tuple[Variable, ...]:
        """덧셈 순전파

        Args:
            xs (Tuple[Variable, ...]): 입력 튜플 (Variable)

        Returns:
            Tuple[Variable, ...]: 출력 튜플 (Variable)
        """
        x0, x1 = xs
        y = x0 + x1
        
        return (y,)