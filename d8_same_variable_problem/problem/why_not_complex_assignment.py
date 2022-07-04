# %%
import numpy as np

# %%
x = np.array(1)
id(x)

# %%
x += x  # 덮어쓰기 (x = x + x)
id(x)  # 똑같은 객체를 가리킴

# %%
x = x + x  # 복사 (새로 생성)
id(x)  # 다른 객체가 나옴!

################
# +=은 메모리 위치가 동일하므로, 값만 덮어쓴 것임
# 복사하지 않고 메모리의 값만 직접 덮어 쓰는 연산을 inplace operation이라 함
# inplace operation은 메모리 효율 측면에서는 좋은 연산임

# %%
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
        # data가 ndarray만 취급
        if data is not None:  # data가 없을 수도(None) 있음
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')
        
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
        # 이전 d4와 비교해 추가된 부분 (y.grad = np.array(1.0) 생략하기 위한 코드)
        if self.grad is None:
            self.grad = np.ones_like(self.data) 
            # self.data와 shape이 같으면서 데이터 타입이 같은(이 부분이 중요) ndarray 생성하여 모든 요소를 1로 채움
            # self.data가 scalar이면 self.grad도 scalar가 됨
        
        # 반복문 버전
        funcs = [self.creator]  # 창조자의 리스트에는 현재 자신에 대한 창조자 1개만 존재
        while funcs:  # funcs가 더 이상 없으면 반복 중단
            f = funcs.pop()     # 창조자(함수)를 가져온다
            
            # 입력을 1개라고 가정한 것임
            #x, y = f.input, f.output  # 창조자(함수)에 대한 입출력을 가져온다 
            #x.grad = f.backward(y.grad) # 창조자에 대해 backward를 호출해 미분값을 계산한다
            
            gys = [output.grad for output in f.outputs]  # outputs에 있는 미분값을 리스트에 담음
            gxs = f.backward(*gys)  # list unpacking -> call backward
            if not isinstance(gxs, tuple): # gxs가 tuple이 아니면 tuple로 packing
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs): # 역전파에서 전파된 미분값을 Variable 인스턴스 변수 grad에 저장
                # gxs와 f.inputs는 서로 대응 관계이므로 zip을 사용
                
                ################ 이 부분이 문제 (problem.py) ####################
                # x.grad = gx
                # 출력 쪽에서 전해지는 미분값을 그대로 대입함
                # 같은 변수를 반복해서 사용하면 전파되는 미분값이 덮어 써짐
                #     /<--- 1 ---- \
                #   x               +' <------ y
                #     \<--- 1 ---- /
                ################################################################
                if x.grad is None:        # 미분 값이 처음으로 설정되면
                    x.grad = gx           # 그대로 출력에게 전해줌
                else:                     # 미분 값이 처음으로 설정되는 것이 아니면
                #    x.grad = x.grad + gx  # 전파되는 미분 값의 합을 사용함
                    x.grad += gx
                # x.grad += gx를 쓰지 않는 이유는 problem/why_not_complex_assignment.py 참고
                
                ################################################################
            
                if x.creator is not None:   # 역방향으로 가기 위해 입력 변수의 창조자가 있다면
                    funcs.append(x.creator) # 반복이 계속 될 수 있도록 창조자 리스트에 추가함
        
    # 내가 만든거 추가
    # Ref: https://shoark7.github.io/programming/python/difference-between-__repr__-vs-__str__
    def __str__(self) -> str:
        """객체를 문자열로 표현하는 함수(내장)

        Returns:
            data (str): data 객체 리턴
        """
        return str(self.data)
# %%
import sys
sys.path.append('..')
from core_simple.calculation import add

# %%
x = Variable(np.array(3))
y = add(x, x)
y.backward()

print(f'y.grad: {y.grad} ({id(y.grad)})')
print(f'x.grad: {x.grad} ({id(x.grad)})')
# y.grad와 x.grad는 같은 객체를 가리키고 있음