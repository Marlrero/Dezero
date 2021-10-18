import numpy as np

# https://www.python.org/dev/peps/pep-0008

class Variable:
    """변수 클래스
    """
    def __init__(self, data: np.ndarray) -> None:
        """변수

        Args:
            data (np.ndarray): Numpy 다차원배열
        """
        self.data = data
        
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