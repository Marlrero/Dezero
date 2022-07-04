import numpy as np

import sys
sys.path.append("..")
from core_simple.variable import Variable
from core_simple.calculation import add

# 같은 변수를 사용해 미분값이 덮어 씌어지는 문제
x = Variable(np.array(3.0))
y = add(x, x)
print(f'y = {y.data}')   # 3.0 + 3.0 = 6.0

y.backward()   # x + x = 2x -> (2x)' = 2
print(f'x.grad = {x.grad}')  # 1 (?) -> Variable.py의 63번째 줄로 변경 후 2

###################################
x = Variable(np.array(3.0))
y = add(add(x, x), x)
print(f'y = {y.data}')   # 3.0 + 3.0 + 3.0 = 9.0

y.backward()   # x + x + x = 3x -> (3x)' = 3
print(f'x.grad = {x.grad}')  # 1 (?) -> Variable.py의 63번째 줄로 변경 후 3

print()
##################################
# 같은 변수를 사용해 다른 계산을 하는 경우 계산이 꼬이는 문제
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(f'x.grad = {x.grad}')   # x + x = 2x -> 2

y = add(add(x, x), x)
y.backward()
print(f'x.grad = {x.grad}')  # x + x + x = 3x -> 3 // 5(?)
# x.grad에 원래 2가 있었고 이를 재사용해 3을 더해서 5가 된 결과가 나온 것이다.
# 즉, 메모리를 절약하기 위해 x 인스턴스를 재사용한 것이다.

print()
# 해결 이후
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(f'x.grad = {x.grad}')   # x + x = 2x -> 2

x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(f'x.grad = {x.grad}')  # x + x + x = 3x -> 3