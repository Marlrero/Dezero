import numpy as np
# https://www.python.org/dev/peps/pep-0008

# Circular reference problem
# https://brunch.co.kr/@mathpresso/18
# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
from typing import TYPE_CHECKING   

if TYPE_CHECKING:  # 단순히 타입체킹을 위해 필요한 모듈이라면!
    import sys
    sys.path.append("..")
    from d2_backward_auto.function import Function


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
        
    """변수 자체에서 창조자(함수)를 가져와 자동 미분 진행 함수
    """
    def backward(self):
        '''
        f = self.creator  # 변수의 창조자(함수)를 가져온다
        if f is not None: # 창조자가 재귀를 거치다가 None을 만나면(사용자가 Variable 인스턴스 바깥에서 직접 생성한 것임) 멈춘다
            x = f.input   # 창조자에 대해 입력을 가져온다
            x.grad = f.backward(self.grad)  # 창조자에 대해 backward를 호출해 미분값을 계산한다
            x.backward()  # 거슬러 올라가기 위해 그 이전에 변수에 대한 backward를 호출한다 (재귀)
        '''
        # 이전 d4와 비교해 추가된 부분 (y.grad = np.array(1.0) 생략하기 위한 코드)
        if self.grad is None:
            self.grad = np.ones_like(self.data) 
            # self.data와 shape이 같으면서 데이터 타입이 같은(이 부분이 중요) ndarray 생성하여 모든 요소를 1로 채움
            # self.data가 scalar이면 self.grad도 scalar가 됨
        
        # 반복문 버전
        funcs = [self.creator]  # 창조자의 리스트에는 현재 자신에 대한 창조자 1개만 존재
        while funcs:  # funcs가 더 이상 없으면 반복 중단
            f = funcs.pop()     # 창조자(함수)를 가져온다
            x, y = f.input, f.output  # 창조자(함수)에 대한 입출력을 가져온다
            x.grad = f.backward(y.grad) # 창조자에 대해 backward를 호출해 미분값을 계산한다
            
            if x.creator is not None:   # 역방향으로 가기 위해 입력 변수의 창조자가 있다면
                funcs.append(x.creator) # 반복이 계속 될 수 있도록 창조자 리스트에 추가함
        
    # 내가 만든거 추가
    # Ref: https://shoark7.github.io/programming/python/difference-between-__repr__-vs-__str__
    def __str__(self) -> str:
        """객체를 문자열로 표현하는 함수(내장)

        Returns:
            data (np.ndarray): data 객체 리턴
        """
        return str(self.data)
    
if __name__ == '__main__':
    from function import square, exp
    
    def main():
        x = Variable(np.array(0.5))
        y = square(exp(square(x)))
        y.backward()
        print(f"y.grad = {y.grad}\nx.grad = {x.grad}")
        
        # Variable 생성 시 np.ndarray만 되는지 확인하는 절차
        a = Variable(1) # 오류가 나지 않음!
                
    main()