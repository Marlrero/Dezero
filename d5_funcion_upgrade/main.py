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