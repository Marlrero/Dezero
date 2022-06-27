# %%
import numpy as np
from function import Function, Square, Exp
from variable import Variable
# %%
x = Variable(np.array(0.5))
f = Square()
y = f(x)
y.data
# %%
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)
# %%
x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)
print(f"y value = {y.data}")

y.grad = np.array(1.0)
y.backward()
print(f"x.grad = {x.grad}")
# %%
def square(x):
    return Square()(x)  # 한줄로 작성 가능
def exp(x):
    return Exp()(x)     # 한줄로 작성 가능
# %%
x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)
print(f"y value = {y.data}")

y.grad = np.array(1.0)
y.backward()
print(f"x.grad = {x.grad}")
# %%
x = Variable(np.array(0.5))
y = square(exp(square(x))) # 이처럼 연속적으로 사용 가능
y.grad = np.array(1.0)
y.backward()
print(f"x.grad = {x.grad}")
# 위와 같이 클래스에서 파이썬 함수로 감싸주게 되면 편함



# %%
# Variable의 data가 np.ndarray만 처리하는지 확인
import numpy as np
from variable import Variable
x = Variable(np.array(1.0))
# %%
x = Variable(None)
# %%
x = Variable(1.0)
# %%
# numpy의 독특한 관례로 잘못될 수도 있음을 보여주는 예
x = np.array([1.0])
y = x ** 2
print(type(x), x.ndim)  # 1차원 ndarray
print(type(y))  # x ** 2을 해도 ndarray
# %%
x = np.array(1.0)
y = x ** 2
print(type(x), x.ndim)  # 0차원 ndarray
print(type(y))          # np.float64가 되버림
# Variable의 데이터는 항상 ndarray여야 한다고 가정하므로 위배될 수 있음



# %%
import numpy as np
np.isscalar(np.float(1.0))
# %%
np.isscalar(2.0)
# %%
np.isscalar(np.array(1.0))
# %%
np.isscalar(np.array([1, 2, 3]))