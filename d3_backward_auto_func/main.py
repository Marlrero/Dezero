import numpy as np

from variable import Variable

import sys
sys.path.append("..")
from d2_backward_auto.function import Square, Exp

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(f"x = {x}\na: {a}\nb: {b}\ny: {y}")
