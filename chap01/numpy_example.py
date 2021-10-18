import numpy as np

x = np.array(1)
print(x.ndim)

x = np.array([1, 2, 3])
print(x.ndim)

x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(x.ndim)

# 배열의 차원(axis, dimension)임에 주의! 벡터 차원 아님!